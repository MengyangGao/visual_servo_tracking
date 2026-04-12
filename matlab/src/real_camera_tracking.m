function result = real_camera_tracking(cfg, mode, sourceMode, runMode)
%REAL_CAMERA_TRACKING Real-camera board following and IBVS demos.

if nargin < 2 || isempty(mode)
    mode = 'follow';
end
if nargin < 3 || isempty(sourceMode)
    sourceMode = 'auto';
end
if nargin < 4 || isempty(runMode)
    runMode = 'auto';
end

mode = lower(string(mode));
sourceMode = lower(string(sourceMode));
runMode = normalize_run_mode(runMode);
isManual = strcmp(runMode, "manual");

switch mode
    case {"follow", "tracking", "t2"}
        outDir = cfg.paths.realFixedTracking;
        stepLimit = cfg.realCamera.followCount;
        stepInterval = cfg.realCamera.followIntervalSec;
        metricLabel = "follow";
    case {"ibvs", "feature", "t3"}
        outDir = cfg.paths.realIbvs;
        stepLimit = cfg.realCamera.ibvsCount;
        stepInterval = cfg.realCamera.ibvsIntervalSec;
        metricLabel = "ibvs";
    otherwise
        error('real_camera_tracking:UnknownMode', 'Unsupported mode "%s".', mode);
end
if isManual
    stepLimit = Inf;
end

if isManual && ~cfg.showFigures
    error('real_camera_tracking:ManualModeRequiresVisibleFigure', 'Manual mode requires visible figures.');
end

ensure_dir(outDir);
ensure_dir(cfg.paths.liveCalibration);

board = charuco_board_asset(cfg);
[cameraParams, cameraParamsPath] = load_camera_params(cfg);
[frameSource, sourceLabel] = local_open_frame_source(cfg, sourceMode, stepLimit, stepInterval);

logPath = fullfile(outDir, sprintf('real_camera_%s_log.txt', metricLabel));
if isfile(logPath)
    delete(logPath);
end
diary(logPath);
logCleanup = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('[RealCamera] Mode: %s\n', metricLabel);
fprintf('[RealCamera] Run mode: %s\n', runMode);
fprintf('[RealCamera] Source: %s\n', sourceLabel);
fprintf('[RealCamera] Camera params: %s\n', cameraParamsPath);
fprintf('[RealCamera] Printable board PNG: %s\n', board.paths.png);
fprintf('[RealCamera] Printable board PDF: %s\n', board.paths.pdf);

robot = cfg.robot;
eeName = cfg.eeName;
ik = inverseKinematics('RigidBodyTree', robot);
weights = cfg.t2.ikWeights;

figureVisible = local_visibility(cfg.showFigures);
fig = figure('Name', sprintf('Real camera - %s', metricLabel), 'Color', 'w', 'Visible', figureVisible);
tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
axWorld = nexttile;
axImage = nexttile;
session = session_controls(fig, runMode, sprintf('Real camera %s', metricLabel));

if session.manual
    session.waitForStart();
    if session.shouldStop()
        close(fig);
        error('real_camera_tracking:StoppedBeforeStart', 'The session was stopped before capture started.');
    end
end

pre_capture_pause(cfg.realCamera.preCaptureDelaySec, 'RealCamera');

if cfg.saveVideos
    videoFile = fullfile(outDir, sprintf('real_camera_%s_tracking.mp4', metricLabel));
    [writer, videoPath] = open_video_writer(videoFile, cfg.videoFrameRate);
    cleanupWriter = onCleanup(@() close(writer)); %#ok<NASGU>
else
    videoPath = '';
end

q = local_initial_configuration(robot, eeName, cfg, ik, weights, metricLabel);
T_ee = getTransform(robot, q, eeName);

timeHistory = zeros(0, 1);
targetHistory = zeros(0, 3);
commandHistory = zeros(0, 3);
eeHistory = zeros(0, 3);
errorHistory = zeros(0, 1);
bboxAreaHistory = zeros(0, 1);
reprojectionHistory = zeros(0, 1);
centroidHistory = zeros(0, 2);
visibilityHistory = false(0, 1);
qHistory = zeros(0, 6);
sourceFrameIds = zeros(0, 1);
featureHistory = nan(0, 8);
desiredFeatureHistory = nan(0, 8);

referenceCenterCamera = [];
referenceCenterPixel = [];
commandPos = local_initial_target(cfg, metricLabel);
virtualCameraT = local_initial_virtual_camera(cfg, metricLabel);
desiredFeatureCache = struct();
samplesProcessed = 0;
historyLimit = cfg.realCamera.historyLimit;
if isempty(historyLimit) || ~isfinite(historyLimit) || historyLimit <= 0
    historyLimit = Inf;
end

while true
    if session.shouldStop()
        break;
    end
    if isfinite(stepLimit) && samplesProcessed >= stepLimit
        break;
    end

    k = samplesProcessed + 1;
    currentTime = (k - 1) * stepInterval;
    timeHistory(k, 1) = currentTime;

    frameInfo = frameSource.readFrame(k);
    sourceFrameIds(k, 1) = frameInfo.frameId;
    rawFrame = frameInfo.frame;

    obs = real_camera_charuco_observation(rawFrame, cameraParams, board);
    if obs.success
        visibilityHistory(k, 1) = true;
        centroidHistory(k, :) = obs.centerPixel;
        bboxAreaHistory(k, 1) = obs.bbox(3) * obs.bbox(4);
        reprojectionHistory(k, 1) = obs.reprojectionError;
        if isempty(referenceCenterCamera)
            referenceCenterCamera = obs.centerCamera;
            referenceCenterPixel = obs.centerPixel;
        end
    else
        visibilityHistory(k, 1) = false;
        centroidHistory(k, :) = [NaN NaN];
        bboxAreaHistory(k, 1) = NaN;
        reprojectionHistory(k, 1) = NaN;
    end

    if strcmp(metricLabel, "follow")
        [targetPos, commandPos, virtualCameraT] = local_update_follow_target( ...
            cfg, obs, commandPos, referenceCenterCamera, referenceCenterPixel, virtualCameraT);
        targetHistory(k, :) = targetPos;
        commandHistory(k, :) = commandPos;
        desiredWorldT = local_downward_pose(commandPos);
        [qNew, solInfo] = ik(eeName, desiredWorldT, weights, q);
        if ~isstruct(solInfo) || ~isfield(solInfo, 'ExitFlag') || solInfo.ExitFlag <= 0
            error('real_camera_tracking:IKFailure', 'IK failed at step %d in follow mode.', k);
        end
        q = local_clamp_to_limits(robot, qNew);
        T_ee = getTransform(robot, q, eeName);
        eeHistory(k, :) = T_ee(1:3, 4).';
        errorHistory(k, 1) = norm(eeHistory(k, 1:2) - targetPos(1:2));

        local_draw_follow_frame(axWorld, axImage, robot, q, eeName, cfg, obs, targetPos, commandPos, k, stepLimit, metricLabel);
    else
        [featureData, desiredFeatureCache] = local_update_ibvs_state(cfg, obs, desiredFeatureCache);
        featureHistory(k, :) = featureData.currentVec.';
        desiredFeatureHistory(k, :) = featureData.desiredVec.';

        if featureData.hasMeasurement
            featureError = featureData.currentVec - featureData.desiredVec;
            L = local_translation_matrix(featureData.currentVec, featureData.depths);
            vCam = -cfg.realCamera.ibvsGain * dls_solve(L, featureError, cfg.t3.interactionDamping);
            virtualCameraT = local_apply_body_translation(virtualCameraT, vCam, stepInterval);
            targetHistory(k, :) = virtualCameraT(1:3, 4).';
            commandHistory(k, :) = virtualCameraT(1:3, 4).';
            [qNew, solInfo] = ik(eeName, virtualCameraT, weights, q);
            if ~isstruct(solInfo) || ~isfield(solInfo, 'ExitFlag') || solInfo.ExitFlag <= 0
                error('real_camera_tracking:IKFailure', 'IK failed at step %d in IBVS mode.', k);
            end
            q = local_clamp_to_limits(robot, qNew);
            T_ee = getTransform(robot, q, eeName);
            eeHistory(k, :) = T_ee(1:3, 4).';
            qHistory(k, :) = q;
            errorHistory(k, 1) = featureData.featureErrorNorm;
        else
            eeHistory(k, :) = T_ee(1:3, 4).';
            qHistory(k, :) = q;
            errorHistory(k, 1) = NaN;
            targetHistory(k, :) = virtualCameraT(1:3, 4).';
            commandHistory(k, :) = virtualCameraT(1:3, 4).';
        end

        local_draw_ibvs_frame(axWorld, axImage, robot, q, eeName, cfg, obs, featureData, desiredFeatureCache, virtualCameraT, k, stepLimit, metricLabel);
    end

    qHistory(k, :) = q;
    drawnow;
    if cfg.saveVideos
        frame = getframe(fig);
        writeVideo(writer, frame);
    end

    samplesProcessed = k;
    if isManual && isfinite(historyLimit) && numel(timeHistory) > historyLimit
        keep = numel(timeHistory) - historyLimit + 1 : numel(timeHistory);
        timeHistory = timeHistory(keep, :);
        targetHistory = targetHistory(keep, :);
        commandHistory = commandHistory(keep, :);
        eeHistory = eeHistory(keep, :);
        errorHistory = errorHistory(keep, :);
        bboxAreaHistory = bboxAreaHistory(keep, :);
        reprojectionHistory = reprojectionHistory(keep, :);
        centroidHistory = centroidHistory(keep, :);
        visibilityHistory = visibilityHistory(keep, :);
        qHistory = qHistory(keep, :);
        sourceFrameIds = sourceFrameIds(keep, :);
        featureHistory = featureHistory(keep, :);
        desiredFeatureHistory = desiredFeatureHistory(keep, :);
    end

    if ~isfield(frameInfo, 'pauseSec') || isempty(frameInfo.pauseSec)
        pause(stepInterval);
    elseif frameInfo.pauseSec > 0
        pause(frameInfo.pauseSec);
    end
end

time = timeHistory;
actualSamples = numel(time);
if actualSamples == 0
    session.cleanup();
    if isgraphics(fig)
        delete(fig);
    end
    error('real_camera_tracking:NoSamples', 'The real-camera session produced no samples.');
end

figurePaths = struct();
if cfg.saveFigures
    switch metricLabel
        case "follow"
            figurePaths = local_save_follow_figures(cfg, figureVisible, outDir, sourceLabel, targetHistory, commandHistory, eeHistory, qHistory, errorHistory, bboxAreaHistory, visibilityHistory, reprojectionHistory, sourceFrameIds);
        otherwise
            figurePaths = local_save_ibvs_figures(cfg, figureVisible, outDir, sourceLabel, errorHistory, featureHistory, desiredFeatureHistory, qHistory, desiredFeatureCache, virtualCameraT, visibilityHistory, sourceFrameIds);
    end
end

session.cleanup();
if isgraphics(fig)
    delete(fig);
end

result = struct();
result.mode = char(mode);
result.runMode = char(runMode);
result.source = char(sourceLabel);
result.cameraParamsPath = cameraParamsPath;
result.board = board;
result.metrics = struct();
result.metrics.samplesProcessed = samplesProcessed;
result.metrics.storedSamples = actualSamples;
lastDetectionIdx = find(visibilityHistory, 1, 'last');
if isempty(lastDetectionIdx)
    result.metrics.detectionStartStep = actualSamples + 1;
    result.metrics.finalDetectionVisible = false;
else
    result.metrics.detectionStartStep = find(visibilityHistory, 1, 'first');
    result.metrics.finalDetectionVisible = visibilityHistory(lastDetectionIdx);
end
result.metrics.meanBoundingBoxArea = mean(bboxAreaHistory, 'omitnan');
result.metrics.meanReprojectionError = mean(reprojectionHistory, 'omitnan');
switch metricLabel
    case "follow"
        lastErrorIdx = find(isfinite(errorHistory), 1, 'last');
        if isempty(lastErrorIdx)
            result.metrics.finalPositionError = NaN;
        else
            result.metrics.finalPositionError = errorHistory(lastErrorIdx);
        end
        result.metrics.rmsePositionError = sqrt(mean(errorHistory .^ 2, 'omitnan'));
        lastEeIdx = find(~isnan(eeHistory(:, 1)), 1, 'last');
        if isempty(lastEeIdx)
            result.metrics.finalEndEffectorPosition = nan(1, 3);
        else
            result.metrics.finalEndEffectorPosition = eeHistory(lastEeIdx, :);
        end
    otherwise
        lastErrorIdx = find(isfinite(errorHistory), 1, 'last');
        if isempty(lastErrorIdx)
            result.metrics.finalFeatureError = NaN;
        else
            result.metrics.finalFeatureError = errorHistory(lastErrorIdx);
        end
        result.metrics.rmseFeatureError = sqrt(mean(errorHistory .^ 2, 'omitnan'));
        lastCameraIdx = find(~isnan(eeHistory(:, 1)), 1, 'last');
        if isempty(lastCameraIdx)
            result.metrics.finalCameraPosition = nan(1, 3);
        else
            result.metrics.finalCameraPosition = eeHistory(lastCameraIdx, :);
        end
end
result.history = struct();
result.history.frameIds = sourceFrameIds;
result.history.time = time;
result.history.target = targetHistory;
result.history.command = commandHistory;
result.history.ee = eeHistory;
result.history.q = qHistory;
result.history.error = errorHistory;
result.history.bboxArea = bboxAreaHistory;
result.history.visible = visibilityHistory;
result.history.reprojectionError = reprojectionHistory;
result.history.features = featureHistory;
result.history.desiredFeatures = desiredFeatureHistory;
result.paths = struct();
result.paths.video = videoPath;
result.paths.figures = figurePaths;
result.paths.outDir = outDir;
result.paths.logPath = logPath;

fprintf('[RealCamera] %s session processed %d samples.\n', upper(metricLabel), samplesProcessed);
fprintf('[RealCamera] Log stored at %s\n', logPath);
end

function [frameSource, sourceLabel] = local_open_frame_source(cfg, sourceMode, requestedCount, stepInterval)
sequenceFiles = local_collect_frame_sequence(cfg);
available = string(webcamlist);
useSequence = strcmp(sourceMode, "files") || (strcmp(sourceMode, "auto") && isempty(available) && ~isempty(sequenceFiles));

if useSequence
    if isempty(sequenceFiles)
        error('real_camera_tracking:NoFrameSequence', 'Requested file sequence source but no matching images were found.');
    end
    sourceLabel = "files";
    frameSource = local_make_sequence_source(sequenceFiles, requestedCount);
else
    sourceLabel = "webcam";
    frameSource = local_make_webcam_source(cfg, requestedCount, stepInterval);
end
end

function files = local_collect_frame_sequence(cfg)
files = {};
folder = cfg.realCamera.frameSequenceDir;
pattern = cfg.realCamera.frameSequencePattern;
listing = dir(fullfile(folder, pattern));
if isempty(listing)
    return;
end

[~, order] = sort(local_extract_sequence_index({listing.name}));
listing = listing(order);
files = cell(numel(listing), 1);
for i = 1:numel(listing)
    files{i} = fullfile(listing(i).folder, listing(i).name);
end
end

function indices = local_extract_sequence_index(names)
indices = zeros(numel(names), 1);
for i = 1:numel(names)
    tokens = regexp(names{i}, '(\d+)', 'match');
    if isempty(tokens)
        indices(i) = i;
    else
        indices(i) = str2double(tokens{end});
    end
end
end

function source = local_make_sequence_source(files, requestedCount)
count = min(requestedCount, numel(files));
source = struct();
source.numFrames = count;
source.readFrame = @readFrame;
    function frameInfo = readFrame(k)
        idx = min(k, numel(files));
        frame = imread(files{idx});
        frameInfo = struct();
        frameInfo.frame = frame;
        frameInfo.frameId = idx;
        frameInfo.pauseSec = 0;
    end
end

function source = local_make_webcam_source(cfg, requestedCount, stepInterval)
available = string(webcamlist);
if isempty(available)
    error('real_camera_tracking:NoCamera', 'No webcam devices were detected.');
end

preferences = string(cfg.realCamera.cameraNamePreference);
cameraName = available(1);
for i = 1:numel(preferences)
    idx = find(strcmpi(available, preferences(i)), 1);
    if ~isempty(idx)
        cameraName = available(idx);
        break;
    end
end

cam = webcam(cameraName);
cleanup = onCleanup(@() delete(cam)); %#ok<NASGU>
source = struct();
source.numFrames = requestedCount;
source.readFrame = @readFrame;
    function frameInfo = readFrame(k)
        frame = snapshot(cam);
        frameInfo = struct();
        frameInfo.frame = frame;
        frameInfo.frameId = k;
        frameInfo.pauseSec = stepInterval;
    end
end

function q = local_initial_configuration(robot, eeName, cfg, ik, weights, mode)
switch mode
    case "follow"
        target = cfg.t2.initialHover;
    otherwise
        target = cfg.t3.initialCameraPosition;
end
desiredT = local_downward_pose(target);
[q, solInfo] = ik(eeName, desiredT, weights, zeros(1, 6));
if ~isstruct(solInfo) || ~isfield(solInfo, 'ExitFlag') || solInfo.ExitFlag <= 0
    error('real_camera_tracking:InitialIKFailure', 'Unable to find an initial robot pose.');
end
q = local_clamp_to_limits(robot, q);
end

function targetPos = local_initial_target(cfg, mode)
switch mode
    case "follow"
        targetPos = cfg.t2.targetPathCenter + [0 0 0];
        targetPos(3) = cfg.t2.hoverHeight;
    otherwise
        targetPos = cfg.t3.desiredCameraPosition;
end
end

function T_wc = local_initial_virtual_camera(cfg, mode)
switch mode
    case "follow"
        T_wc = local_downward_pose(cfg.t2.initialHover);
    otherwise
        T_wc = local_downward_pose(cfg.t3.initialCameraPosition);
end
end

function [targetPos, commandPos, virtualCameraT] = local_update_follow_target(cfg, obs, commandPos, referenceCenterCamera, referenceCenterPixel, virtualCameraT)
targetPos = commandPos;
if obs.success
    proxy = cfg.realCamera.followProxyScale;
    if ~isempty(referenceCenterCamera) && all(isfinite(referenceCenterCamera)) && all(isfinite(obs.centerCamera))
        delta = obs.centerCamera(1:2) - referenceCenterCamera(1:2);
        targetPos = [cfg.t2.targetPathCenter(1) + proxy * delta(1), ...
                     cfg.t2.targetPathCenter(2) + proxy * delta(2), ...
                     cfg.t2.hoverHeight];
    elseif ~isempty(referenceCenterPixel) && all(isfinite(referenceCenterPixel))
        frameCenter = [size(obs.undistortedFrame, 2) / 2, size(obs.undistortedFrame, 1) / 2];
        deltaPx = (obs.centerPixel - referenceCenterPixel) ./ frameCenter;
        targetPos = [cfg.t2.targetPathCenter(1) + proxy * deltaPx(1), ...
                     cfg.t2.targetPathCenter(2) + proxy * deltaPx(2), ...
                     cfg.t2.hoverHeight];
    else
        targetPos = [cfg.t2.targetPathCenter(1:2), cfg.t2.hoverHeight];
    end
    commandPos = commandPos + cfg.t2.commandAlpha * (targetPos - commandPos);
    virtualCameraT = local_downward_pose(commandPos);
elseif isempty(referenceCenterCamera)
    targetPos = [cfg.t2.targetPathCenter(1:2), cfg.t2.hoverHeight];
end
end

function [featureData, desiredFeatureCache] = local_update_ibvs_state(cfg, obs, desiredFeatureCache)
featureData = struct();
featureData.hasMeasurement = false;
featureData.featureErrorNorm = NaN;
featureData.currentCorners = nan(4, 2);
featureData.desiredCorners = nan(4, 2);
featureData.currentVec = nan(8, 1);
featureData.desiredVec = nan(8, 1);
featureData.depths = nan(4, 1);

if ~obs.success
    return;
end

featureData.hasMeasurement = true;
if isempty(desiredFeatureCache) || ~isfield(desiredFeatureCache, 'cornerPixels') || isempty(desiredFeatureCache.cornerPixels)
    desiredFeatureCache.cornerPixels = local_desired_square_from_measurement(obs, cfg);
end

currentCorners = obs.bboxCorners;
desiredCorners = desiredFeatureCache.cornerPixels;
featureData.currentCorners = currentCorners;
featureData.desiredCorners = desiredCorners;
featureData.currentVec = local_normalized_feature_vector(currentCorners, obs.poseCameraParams);
featureData.desiredVec = local_normalized_feature_vector(desiredCorners, obs.poseCameraParams);
featureData.featureErrorNorm = norm(featureData.currentVec - featureData.desiredVec);

if ~isempty(obs.centerCamera) && all(isfinite(obs.centerCamera))
    targetCenter = obs.centerCamera;
else
    targetCenter = [cfg.t3.desiredCameraPosition(1), cfg.t3.desiredCameraPosition(2), cfg.t3.desiredCameraPosition(3)];
end
featureData.depths = max(targetCenter(3), 0.45) * ones(4, 1);
end

function corners = local_desired_square_from_measurement(obs, cfg)
frameSize = size(obs.undistortedFrame);
centerPixel = [frameSize(2) / 2, frameSize(1) / 2];
if any(~isfinite(centerPixel))
    centerPixel = [frameSize(2) / 2, frameSize(1) / 2];
end
if isempty(obs.bbox) || any(~isfinite(obs.bbox))
    side = 0.35 * min(frameSize(1), frameSize(2));
else
    side = max(80, 0.90 * max(obs.bbox(3), obs.bbox(4)));
end
half = side / 2;
corners = [centerPixel(1) - half, centerPixel(2) - half;
           centerPixel(1) + half, centerPixel(2) - half;
           centerPixel(1) + half, centerPixel(2) + half;
           centerPixel(1) - half, centerPixel(2) + half];
end

function vec = local_feature_vector_from_corners(corners)
vec = reshape(corners.', [], 1);
end

function vec = local_normalized_feature_vector(corners, cameraParams)
[~, fx, fy, cx, cy] = extract_intrinsics(cameraParams);
normCorners = [ ...
    (corners(:, 1) - cx) ./ fx, ...
    (corners(:, 2) - cy) ./ fy];
vec = reshape(normCorners.', [], 1);
end

function L = local_translation_matrix(s, depths)
numPts = numel(s) / 2;
L = zeros(2 * numPts, 3);
for i = 1:numPts
    x = s(2 * i - 1);
    y = s(2 * i);
    Z = depths(i);
    L(2 * i - 1:2 * i, :) = [ ...
        -1 / Z, 0, x / Z;
        0, -1 / Z, y / Z];
end
end

function T_new = local_apply_body_translation(T_current, vCam, dt)
T_new = T_current;
T_new(1:3, 4) = T_current(1:3, 4) + T_current(1:3, 1:3) * (vCam(:) * dt);
end

function local_draw_follow_frame(axWorld, axImage, robot, q, eeName, cfg, obs, targetPos, commandPos, stepIdx, totalSteps, metricLabel)
cla(axWorld);
hold(axWorld, 'on');
axis(axWorld, [cfg.scene.tableX, cfg.scene.tableY, -0.02, 0.95]);
axis(axWorld, 'equal');
xlabel(axWorld, 'X (m)');
ylabel(axWorld, 'Y (m)');
zlabel(axWorld, 'Z (m)');
if isfinite(totalSteps)
    title(axWorld, sprintf('Real camera follow step %d/%d', stepIdx, totalSteps));
else
    title(axWorld, sprintf('Real camera follow step %d', stepIdx));
end
patch(axWorld, 'XData', [cfg.scene.tableX(1), cfg.scene.tableX(2), cfg.scene.tableX(2), cfg.scene.tableX(1)], ...
    'YData', [cfg.scene.tableY(1), cfg.scene.tableY(1), cfg.scene.tableY(2), cfg.scene.tableY(2)], ...
    'ZData', [cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ], ...
    'FaceColor', [0.95 0.93 0.89], 'FaceAlpha', 0.7, 'EdgeColor', [0.35 0.3 0.25]);
plot3(axWorld, targetPos(1), targetPos(2), targetPos(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(axWorld, commandPos(1), commandPos(2), commandPos(3), 'bx', 'MarkerSize', 10, 'LineWidth', 1.5);
show(robot, q, 'Parent', axWorld, 'Frames', 'off', 'PreservePlot', false, 'Visuals', 'on');
T_ee = getTransform(robot, q, eeName);
draw_camera_frame(axWorld, T_ee, 0.055, 'EE', 1.4);
apply_world_view(axWorld, cfg, cfg.scene);
add_status_badge(axWorld, sprintf('FOLLOW | %s', ternary(obs.success, 'TRACKING', 'SEARCHING')), cfg.visualization.worldBadgePosition, [0.10 0.18 0.30], [1 1 1]);
apply_axes_theme(axWorld, 'plot');

cla(axImage);
img = obs.undistortedFrame;
imshow(img, 'Parent', axImage);
hold(axImage, 'on');
title(axImage, sprintf('Detected ChArUco board%s', ternary(obs.success, '', ' (not found)')));
if obs.success
    plot(axImage, obs.points(:, 1), obs.points(:, 2), 'g.', 'MarkerSize', 10);
    rectangle(axImage, 'Position', obs.bbox, 'EdgeColor', [0 1 0], 'LineWidth', 1.8);
    plot(axImage, obs.centerPixel(1), obs.centerPixel(2), 'r+', 'MarkerSize', 10, 'LineWidth', 1.5);
end
add_status_badge(axImage, ternary(obs.success, 'DETECTED', 'SEARCHING'), cfg.visualization.imageBadgePosition, ternary(obs.success, [0.15 0.58 0.28], [0.30 0.43 0.60]), [1 1 1]);
apply_axes_theme(axImage, 'image');
hold(axImage, 'off');
end

function local_draw_ibvs_frame(axWorld, axImage, robot, q, eeName, cfg, obs, featureData, desiredFeatureCache, virtualCameraT, stepIdx, totalSteps, metricLabel)
cla(axWorld);
hold(axWorld, 'on');
axis(axWorld, [cfg.scene.tableX, cfg.scene.tableY, -0.02, 0.95]);
axis(axWorld, 'equal');
xlabel(axWorld, 'X (m)');
ylabel(axWorld, 'Y (m)');
zlabel(axWorld, 'Z (m)');
if isfinite(totalSteps)
    title(axWorld, sprintf('Real camera IBVS step %d/%d', stepIdx, totalSteps));
else
    title(axWorld, sprintf('Real camera IBVS step %d', stepIdx));
end
patch(axWorld, 'XData', [cfg.scene.tableX(1), cfg.scene.tableX(2), cfg.scene.tableX(2), cfg.scene.tableX(1)], ...
    'YData', [cfg.scene.tableY(1), cfg.scene.tableY(1), cfg.scene.tableY(2), cfg.scene.tableY(2)], ...
    'ZData', [cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ], ...
    'FaceColor', [0.95 0.93 0.89], 'FaceAlpha', 0.7, 'EdgeColor', [0.35 0.3 0.25]);
plot3(axWorld, virtualCameraT(1, 4), virtualCameraT(2, 4), virtualCameraT(3, 4), 'bp', 'MarkerSize', 14, 'MarkerFaceColor', 'b');
show(robot, q, 'Parent', axWorld, 'Frames', 'off', 'PreservePlot', false, 'Visuals', 'on');
draw_camera_frame(axWorld, virtualCameraT, 0.055, 'cam', 1.4);
apply_world_view(axWorld, cfg, cfg.scene);
add_status_badge(axWorld, sprintf('IBVS | %s', ternary(featureData.hasMeasurement, 'TRACKING', 'SEARCHING')), cfg.visualization.worldBadgePosition, [0.10 0.18 0.30], [1 1 1]);
apply_axes_theme(axWorld, 'plot');

cla(axImage);
img = obs.undistortedFrame;
imshow(img, 'Parent', axImage);
hold(axImage, 'on');
title(axImage, sprintf('IBVS image (feature error = %.4f)', featureData.featureErrorNorm));
if obs.success
    plot(axImage, featureData.currentCorners(:, 1), featureData.currentCorners(:, 2), 'ro', 'MarkerSize', 6, 'LineWidth', 1.3);
    plot(axImage, featureData.desiredCorners(:, 1), featureData.desiredCorners(:, 2), 'gx', 'MarkerSize', 9, 'LineWidth', 1.5);
    for i = 1:size(featureData.currentCorners, 1)
        plot(axImage, [featureData.currentCorners(i, 1), featureData.desiredCorners(i, 1)], ...
            [featureData.currentCorners(i, 2), featureData.desiredCorners(i, 2)], 'y-', 'LineWidth', 0.9);
    end
end
add_status_badge(axImage, ternary(obs.success, 'FEATURES LOCKED', 'SEARCHING'), cfg.visualization.imageBadgePosition, ternary(obs.success, [0.15 0.58 0.28], [0.30 0.43 0.60]), [1 1 1]);
apply_axes_theme(axImage, 'image');
hold(axImage, 'off');
end

function figurePaths = local_save_follow_figures(cfg, figureVisible, outDir, sourceLabel, targetHistory, commandHistory, eeHistory, qHistory, errorHistory, bboxAreaHistory, visibilityHistory, reprojectionHistory, sourceFrameIds)
figurePaths = struct();

trajFig = figure('Name', 'Real camera follow trajectory', 'Color', 'w', 'Visible', figureVisible, 'Position', [120 90 980 1120]);
tiledlayout(trajFig, 3, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile;
hold(ax, 'on');
plot(ax, targetHistory(:, 1), targetHistory(:, 2), '-g', 'LineWidth', 1.8);
plot(ax, commandHistory(:, 1), commandHistory(:, 2), '-b', 'LineWidth', 1.3);
plot(ax, eeHistory(:, 1), eeHistory(:, 2), '-r', 'LineWidth', 1.3);
scatter(ax, targetHistory(1, 1), targetHistory(1, 2), 60, [0.2 0.2 0.2], 'filled');
scatter(ax, targetHistory(end, 1), targetHistory(end, 2), 70, 'g', 'filled');
scatter(ax, eeHistory(end, 1), eeHistory(end, 2), 70, 'r', 'filled');
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
title(ax, sprintf('Target, command, and end-effector path (%s)', sourceLabel));
legend(ax, {'Target path', 'Command path', 'End effector path'}, 'Location', 'best');
axis(ax, 'equal');
grid(ax, 'on');
apply_axes_theme(ax, 'plot');

ax = nexttile;
hold(ax, 'on');
plot(ax, targetHistory(:, 1), '-g', 'LineWidth', 1.5);
plot(ax, commandHistory(:, 1), '-b', 'LineWidth', 1.2);
plot(ax, eeHistory(:, 1), '-r', 'LineWidth', 1.2);
xlabel(ax, 'Frame');
ylabel(ax, 'X position (m)');
title(ax, 'X position over time');
legend(ax, {'Target X', 'Command X', 'End effector X'}, 'Location', 'best');
grid(ax, 'on');
apply_axes_theme(ax, 'plot');

ax = nexttile;
hold(ax, 'on');
plot(ax, targetHistory(:, 2), '-g', 'LineWidth', 1.5);
plot(ax, commandHistory(:, 2), '-b', 'LineWidth', 1.2);
plot(ax, eeHistory(:, 2), '-r', 'LineWidth', 1.2);
xlabel(ax, 'Frame');
ylabel(ax, 'Y position (m)');
title(ax, 'Y position over time');
legend(ax, {'Target Y', 'Command Y', 'End effector Y'}, 'Location', 'best');
grid(ax, 'on');
apply_axes_theme(ax, 'plot');
figurePaths.trajectoryPath = fullfile(outDir, 'real_camera_follow_trajectory.png');
save_figure(trajFig, figurePaths.trajectoryPath);
close(trajFig);

errorFig = figure('Name', 'Real camera follow error', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', errorFig);
plot(ax, errorHistory, 'm-', 'LineWidth', 1.5);
xlabel(ax, 'Frame');
ylabel(ax, 'Position error (m)');
title(ax, 'Follow error');
apply_axes_theme(ax, 'plot');
figurePaths.errorPath = fullfile(outDir, 'real_camera_follow_error.png');
save_figure(errorFig, figurePaths.errorPath);
close(errorFig);

jointFig = figure('Name', 'Real camera follow joints', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', jointFig);
plot(ax, qHistory, 'LineWidth', 1.0);
xlabel(ax, 'Frame');
ylabel(ax, 'Joint angle (rad)');
title(ax, 'Joint motion');
legend(ax, compose('q%d', 1:6), 'Location', 'eastoutside');
apply_axes_theme(ax, 'plot');
figurePaths.jointPath = fullfile(outDir, 'real_camera_follow_joints.png');
save_figure(jointFig, figurePaths.jointPath);
close(jointFig);

supportFig = figure('Name', 'Real camera follow support', 'Color', 'w', 'Visible', figureVisible);
tiledlayout(supportFig, 2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile;
yyaxis(ax1, 'left');
plot(ax1, sourceFrameIds, bboxAreaHistory, '-o', 'LineWidth', 1.2);
ylabel(ax1, 'BBox area (px^2)');
yyaxis(ax1, 'right');
plot(ax1, sourceFrameIds, double(visibilityHistory), '-s', 'LineWidth', 1.0);
ylabel(ax1, 'Detection visible');
xlabel(ax1, 'Frame ID');
title(ax1, 'Detection support and visibility');
ax2 = nexttile;
plot(ax2, reprojectionHistory, '-^', 'LineWidth', 1.1);
xlabel(ax2, 'Frame');
ylabel(ax2, 'Reprojection error (px)');
title(ax2, 'Per-frame reprojection error');
apply_axes_theme(ax1, 'plot');
apply_axes_theme(ax2, 'plot');
figurePaths.supportPath = fullfile(outDir, 'real_camera_follow_support.png');
save_figure(supportFig, figurePaths.supportPath);
close(supportFig);
end

function figurePaths = local_save_ibvs_figures(cfg, figureVisible, outDir, sourceLabel, errorHistory, featureHistory, desiredFeatureHistory, qHistory, desiredFeatureCache, virtualCameraT, visibilityHistory, sourceFrameIds)
figurePaths = struct();

errorFig = figure('Name', 'Real camera IBVS error', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', errorFig);
plot(ax, errorHistory, 'm-', 'LineWidth', 1.5);
xlabel(ax, 'Frame');
ylabel(ax, 'Feature error norm');
title(ax, sprintf('IBVS feature error (%s)', sourceLabel));
apply_axes_theme(ax, 'plot');
figurePaths.errorPath = fullfile(outDir, 'real_camera_ibvs_error.png');
save_figure(errorFig, figurePaths.errorPath);
close(errorFig);

featureFig = figure('Name', 'Real camera IBVS features', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', featureFig);
plot(ax, featureHistory, 'LineWidth', 1.0);
hold(ax, 'on');
plot(ax, desiredFeatureHistory, '--', 'LineWidth', 1.2);
hold(ax, 'off');
xlabel(ax, 'Frame');
ylabel(ax, 'Normalized image feature');
title(ax, 'Normalized image features and desired targets');
legend(ax, [compose('s%d', 1:size(featureHistory, 2)), compose('s%d*', 1:size(desiredFeatureHistory, 2))], 'Location', 'eastoutside');
apply_axes_theme(ax, 'plot');
figurePaths.featurePath = fullfile(outDir, 'real_camera_ibvs_features.png');
save_figure(featureFig, figurePaths.featurePath);
close(featureFig);

jointFig = figure('Name', 'Real camera IBVS joints', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', jointFig);
plot(ax, qHistory, 'LineWidth', 1.0);
xlabel(ax, 'Frame');
ylabel(ax, 'Joint angle (rad)');
title(ax, 'Joint motion');
legend(ax, compose('q%d', 1:6), 'Location', 'eastoutside');
apply_axes_theme(ax, 'plot');
figurePaths.jointPath = fullfile(outDir, 'real_camera_ibvs_joints.png');
save_figure(jointFig, figurePaths.jointPath);
close(jointFig);

depthFig = figure('Name', 'Real camera IBVS support', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', depthFig);
plot(ax, double(visibilityHistory), '-o', 'LineWidth', 1.2);
xlabel(ax, 'Frame');
ylabel(ax, 'Detected');
title(ax, sprintf('Detection status and virtual camera pose: [%.3f %.3f %.3f]', virtualCameraT(1,4), virtualCameraT(2,4), virtualCameraT(3,4)));
apply_axes_theme(ax, 'plot');
figurePaths.depthPath = fullfile(outDir, 'real_camera_ibvs_support.png');
save_figure(depthFig, figurePaths.depthPath);
close(depthFig);
end

function T_wc = local_downward_pose(position)
position = position(:);
R_wc = [1 0 0;
        0 -1 0;
        0 0 -1];
T_wc = eye(4);
T_wc(1:3, 1:3) = R_wc;
T_wc(1:3, 4) = position;
end

function q = local_clamp_to_limits(robot, q)
lower = -inf(1, numel(q));
upper = inf(1, numel(q));
for i = 1:numel(robot.Bodies)
    joint = robot.Bodies{i}.Joint;
    if ~strcmp(joint.Type, 'fixed') && numel(joint.PositionLimits) == 2
        idx = i;
        if idx >= 1 && idx <= numel(q)
            lower(idx) = joint.PositionLimits(1);
            upper(idx) = joint.PositionLimits(2);
        end
    end
end
q = min(max(q, lower), upper);
end

function y = ternary(cond, a, b)
if cond
    y = a;
else
    y = b;
end
end

function out = local_visibility(flag)
if flag
    out = 'on';
else
    out = 'off';
end
end
