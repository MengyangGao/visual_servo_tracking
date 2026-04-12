function result = t2_position_tracking(cfg, mode, runMode)
%SIMULATION_T2_POSITION_TRACKING Position-based tracking with fixed or eye-in-hand camera.

if nargin < 2
    mode = 'fixed';
end
mode = lower(string(mode));
switch mode
    case {"fixed", "fixed-camera", "overhead"}
        modeLabel = "fixed_camera";
    case {"eye", "eye-in-hand", "eyeinhand"}
        modeLabel = "eye_in_hand";
    otherwise
        error('t2_position_tracking:UnknownMode', 'Unsupported mode "%s".', mode);
end

if nargin < 3 || isempty(runMode)
    runMode = 'auto';
end
runMode = normalize_run_mode(runMode);
isManual = strcmp(runMode, "manual");

switch modeLabel
    case "fixed_camera"
        outDir = cfg.paths.fixedTracking;
    otherwise
        outDir = cfg.paths.eyeTracking;
end
ensure_dir(outDir);

robot = cfg.robot;
eeName = cfg.eeName;
ik = inverseKinematics('RigidBodyTree', robot);
weights = cfg.t2.ikWeights;

if modeLabel == "fixed_camera"
    T_wc_fixed = lookat_tform(cfg.fixedCamera.position, cfg.fixedCamera.target, cfg.fixedCamera.upVector);
    cameraModel = cfg.camera;
else
    T_wc_fixed = [];
    cameraModel = cfg.t2.eyeInHandCamera;
end

q = local_initial_configuration(robot, eeName, cfg.t2.initialHover, cfg.scene.tableZ, cfg.t2.hoverHeight, weights, ik);
eePos0 = getTransform(robot, q, eeName);
eePos0 = eePos0(1:3, 4).';
commandPos = [eePos0(1:2), cfg.t2.hoverHeight];
stepLimit = cfg.t2.numSteps;
if isManual
    stepLimit = Inf;
end

state = struct();
state.detectionVisible = false;
state.motionStarted = false;
state.detectionVisibleStep = NaN;
state.motionStartStep = NaN;
state.motionStartTime = NaN;

figureVisible = local_visibility(cfg.showFigures);
fig = figure('Name', sprintf('Simulation - T2 %s', modeLabel), 'Color', 'w', 'Visible', figureVisible);
tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
axWorld = nexttile;
axCam = nexttile;

if strcmp(runMode, "manual") && ~cfg.showFigures
    error('t2_position_tracking:ManualModeRequiresVisibleFigure', 'Manual mode requires visible figures.');
end
session = session_controls(fig, runMode, sprintf('T2 %s', strrep(char(modeLabel), '_', ' ')));
if session.manual
    session.waitForStart();
    if session.shouldStop()
        close(fig);
        error('t2_position_tracking:StoppedBeforeStart', 'The session was stopped before tracking started.');
    end
end

if cfg.saveVideos
    videoFile = fullfile(outDir, sprintf('t2_%s_tracking.mp4', modeLabel));
    [writer, videoPath] = open_video_writer(videoFile, cfg.videoFrameRate);
    cleanup = onCleanup(@() close(writer)); %#ok<NASGU>
else
    videoPath = '';
end

timeHistory = zeros(0, 1);
targetHistory = zeros(0, 3);
estimateHistory = zeros(0, 3);
commandHistory = zeros(0, 3);
eeHistory = zeros(0, 3);
qHistory = zeros(0, 6);
errorHistory = zeros(0, 1);
detectionArea = zeros(0, 1);
detectionVisibleHistory = false(0, 1);
motionHistory = false(0, 1);

k = 0;
while true
    if session.shouldStop()
        break;
    end
    if isfinite(stepLimit) && k >= stepLimit
        break;
    end

    k = k + 1;
    currentTime = (k - 1) * cfg.t2.dt;
    timeHistory(k, 1) = currentTime;
    motionActive = state.motionStarted;
    targetPos = local_target_position(cfg, currentTime, motionActive, state.motionStartTime);
    targetHistory(k, :) = targetPos;

    scene = local_t2_scene(cfg, targetPos);

    switch modeLabel
        case "fixed_camera"
            T_wc = T_wc_fixed;
        otherwise
            T_wc = getTransform(robot, q, eeName);
    end

    [img, ~] = render_camera_view(T_wc, cameraModel, scene);
    detection = detect_target(img, cfg.t2.targetColorName);
    detectionArea(k, 1) = detection.area;

    if detection.success
        targetEstimate = backproject_to_plane(detection.centroid, T_wc, cameraModel, cfg.scene.tableZ);
        estimateHistory(k, :) = targetEstimate;
    else
        if ~state.detectionVisible
            targetEstimate = [cfg.t2.targetPathCenter(1), cfg.t2.targetPathCenter(2), cfg.scene.tableZ];
            estimateHistory(k, :) = targetEstimate;
        else
            error('t2_position_tracking:DetectionLost', 'Target detection failed after acquisition at step %d in mode %s.', k, modeLabel);
        end
    end

    if ~state.detectionVisible
        commandPos(1:2) = commandPos(1:2) + cfg.t2.commandAlpha * (cfg.t2.targetPathCenter(1:2) - commandPos(1:2));
        commandPos(3) = cfg.t2.hoverHeight;
    else
        commandPos(1:2) = commandPos(1:2) + cfg.t2.commandAlpha * (targetEstimate(1:2) - commandPos(1:2));
        commandPos(3) = cfg.t2.hoverHeight;
    end
    commandHistory(k, :) = commandPos;

    desiredT = local_downward_pose(commandPos);
    [qNew, solInfo] = ik(eeName, desiredT, weights, q);
    if ~isstruct(solInfo) || ~isfield(solInfo, 'ExitFlag') || solInfo.ExitFlag <= 0
        error('t2_position_tracking:IKFailure', 'IK failed at step %d in mode %s.', k, modeLabel);
    end
    q = local_clamp_to_limits(robot, qNew);

    T_ee = getTransform(robot, q, eeName);
    eeHistory(k, :) = T_ee(1:3, 4).';
    qHistory(k, :) = q;

    if ~state.detectionVisible && k >= cfg.t2.staticLeadInSteps && detection.success
        state.detectionVisible = true;
        state.detectionVisibleStep = k;
    end

    if state.detectionVisible && ~state.motionStarted
        if k >= state.detectionVisibleStep + cfg.t2.detectionHoldSteps
            state.motionStarted = true;
            state.motionStartStep = k + 1;
            state.motionStartTime = currentTime + cfg.t2.dt;
        end
    end

    motionActive = state.motionStarted;
    detectionVisible = state.detectionVisible;
    detectionVisibleHistory(k, 1) = detectionVisible;
    motionHistory(k, 1) = motionActive;
    errorHistory(k, 1) = norm(targetEstimate(1:2) - targetPos(1:2));

    local_draw_t2_frame(axWorld, axCam, robot, q, eeName, cfg, scene, img, detection, targetEstimate, commandPos, T_wc, modeLabel, detectionVisible, motionActive, k, stepLimit);
    drawnow;
    if cfg.saveVideos
        frame = getframe(fig);
        writeVideo(writer, frame);
    end
end

time = timeHistory;
actualSteps = numel(time);
if actualSteps == 0
    error('t2_position_tracking:NoSamples', 'The T2 session produced no samples.');
end

session.cleanup();
if isgraphics(fig)
    delete(fig);
end

detectionStartIdx = find(detectionVisibleHistory, 1, 'first');
if isempty(detectionStartIdx)
    detectionStartIdx = actualSteps + 1;
end
motionStartIdx = find(motionHistory, 1, 'first');
if isempty(motionStartIdx)
    motionStartIdx = actualSteps + 1;
end

figurePaths = struct('trajectoryPath', '', 'errorPath', '', 'jointPath', '', 'supportPath', '');
if cfg.saveFigures
    figurePaths = local_save_t2_figures(cfg, figureVisible, outDir, modeLabel, time, targetHistory, estimateHistory, eeHistory, qHistory, errorHistory, detectionArea, detectionVisibleHistory, motionHistory, detectionStartIdx, motionStartIdx);
end

result = struct();
result.mode = char(modeLabel);
result.runMode = char(runMode);
result.metrics = struct();
result.metrics.finalPositionError = errorHistory(end);
result.metrics.rmsePositionError = sqrt(mean(errorHistory .^ 2));
result.metrics.finalEndEffectorError = norm(eeHistory(end, 1:2) - targetHistory(end, 1:2));
result.metrics.meanDetectionArea = mean(detectionArea);
result.metrics.detectionStartStep = detectionStartIdx;
result.metrics.motionStartStep = state.motionStartStep;
result.metrics.motionStartIndex = motionStartIdx;
result.metrics.finalCommandPosition = commandHistory(end, :);
result.metrics.finalEndEffectorPosition = eeHistory(end, :);

result.history = struct();
result.history.time = time;
result.history.target = targetHistory;
result.history.estimate = estimateHistory;
result.history.command = commandHistory;
result.history.ee = eeHistory;
result.history.q = qHistory;
result.history.error = errorHistory;
result.history.detectionArea = detectionArea;
result.history.detectionVisible = detectionVisibleHistory;
result.history.motionStarted = motionHistory;

result.paths = struct();
result.paths.figure = figurePaths.trajectoryPath;
result.paths.figures = figurePaths;
result.paths.video = videoPath;
result.paths.outDir = outDir;
end

function scene = local_t2_scene(cfg, targetPos)
scene = struct();
scene.backgroundColor = cfg.scene.backgroundColor;
scene.objects = struct([]);

scene.objects(1).kind = 'sphere';
scene.objects(1).name = 'target';
scene.objects(1).position = targetPos;
scene.objects(1).radius = cfg.t2.targetRadius;
scene.objects(1).color = uint8([220 50 50]);

for i = 1:numel(cfg.t2.distractors)
    d = cfg.t2.distractors(i);
    scene.objects(end + 1).kind = 'sphere'; %#ok<AGROW>
    scene.objects(end).name = d.name;
    scene.objects(end).position = d.position;
    scene.objects(end).radius = d.radius;
    scene.objects(end).color = d.color;
end
end

function targetPos = local_target_position(cfg, currentTime, motionActive, motionStartTime)
center = [cfg.t2.targetPathCenter(1), cfg.t2.targetPathCenter(2), cfg.scene.tableZ];
if ~motionActive || isnan(motionStartTime)
    targetPos = center;
    return;
end

moveStartTime = motionStartTime;
moveTime = max(currentTime - moveStartTime, 0);
moveDuration = max(cfg.t2.motionDurationSec - moveStartTime, cfg.t2.dt);
progress = min(moveTime / moveDuration, 1);
ease = 0.5 - 0.5 * cos(pi * progress);
theta = 2 * pi * progress;
delta = [ ...
    cfg.t2.targetPathRadius(1) * sin(theta) + cfg.t2.targetPathWiggle(1) * sin(2.4 * theta), ...
    cfg.t2.targetPathRadius(2) * sin(1.3 * theta - cfg.t2.targetPathPhase) + cfg.t2.targetPathWiggle(2) * sin(3.1 * theta)];
targetPos = [center(1) + ease * delta(1), center(2) + ease * delta(2), center(3)];
end

function q0 = local_initial_configuration(robot, eeName, hoverPoint, tableZ, hoverHeight, weights, ik)
desiredT = local_downward_pose(hoverPoint);
[q0, solInfo] = ik(eeName, desiredT, weights, zeros(1, 6));
if ~isstruct(solInfo) || ~isfield(solInfo, 'ExitFlag') || solInfo.ExitFlag <= 0
    fallbackPose = local_downward_pose([0.28 -0.14 hoverHeight]);
    [q0, solInfo] = ik(eeName, fallbackPose, weights, zeros(1, 6));
    if ~isstruct(solInfo) || ~isfield(solInfo, 'ExitFlag') || solInfo.ExitFlag <= 0
        error('t2_position_tracking:InitialIKFailure', 'Unable to find an initial robot pose.');
    end
end
q0 = local_clamp_to_limits(robot, q0);
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

function local_draw_t2_frame(axWorld, axCam, robot, q, eeName, cfg, scene, img, detection, targetEstimate, commandPos, T_wc, modeLabel, detectionVisible, motionActive, stepIdx, totalSteps)
displayModeLabel = strrep(char(modeLabel), '_', ' ');
if ~detectionVisible
    stageLabel = 'approach';
elseif motionActive
    stageLabel = 'tracking';
else
    stageLabel = 'detected';
end

cla(axWorld);
hold(axWorld, 'on');
axis(axWorld, [cfg.scene.tableX, cfg.scene.tableY, -0.02, 0.85]);
axis(axWorld, 'equal');
xlabel(axWorld, 'X (m)');
ylabel(axWorld, 'Y (m)');
zlabel(axWorld, 'Z (m)');
if isfinite(totalSteps)
    title(axWorld, sprintf('T2 %s step %d/%d (%s)', displayModeLabel, stepIdx, totalSteps, stageLabel));
else
    title(axWorld, sprintf('T2 %s step %d (%s)', displayModeLabel, stepIdx, stageLabel));
end

patch(axWorld, 'XData', [cfg.scene.tableX(1), cfg.scene.tableX(2), cfg.scene.tableX(2), cfg.scene.tableX(1)], ...
    'YData', [cfg.scene.tableY(1), cfg.scene.tableY(1), cfg.scene.tableY(2), cfg.scene.tableY(2)], ...
    'ZData', [cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ], ...
    'FaceColor', [0.95 0.93 0.89], 'FaceAlpha', 0.72, 'EdgeColor', [0.35 0.3 0.25]);

for i = 1:numel(scene.objects)
    obj = scene.objects(i);
    if strcmp(obj.name, 'target')
        scatter3(axWorld, obj.position(1), obj.position(2), obj.position(3), 180, double(obj.color) / 255, 'filled', 'MarkerEdgeColor', 'k');
        text(axWorld, obj.position(1), obj.position(2), obj.position(3) + 0.03, 'target', 'Color', [0.5 0 0], 'FontWeight', 'bold');
    else
        scatter3(axWorld, obj.position(1), obj.position(2), obj.position(3), 120, double(obj.color) / 255, 'filled', 'MarkerEdgeColor', 'k');
    end
end

plot3(axWorld, targetEstimate(1), targetEstimate(2), cfg.scene.tableZ, 'rx', 'MarkerSize', 11, 'LineWidth', 2);
plot3(axWorld, commandPos(1), commandPos(2), commandPos(3), 'bp', 'MarkerSize', 14, 'MarkerFaceColor', 'b');
show(robot, q, 'Parent', axWorld, 'Frames', 'off', 'PreservePlot', false, 'Visuals', 'on');
plot3(axWorld, cfg.t2.targetPathCenter(1), cfg.t2.targetPathCenter(2), cfg.scene.tableZ, 'k.', 'MarkerSize', 12);
draw_camera_frame(axWorld, T_wc, 0.055, 'cam', 1.5);
apply_world_view(axWorld, cfg, cfg.scene);
add_status_badge(axWorld, sprintf('%s | %s', upper(displayModeLabel), upper(stageLabel)), cfg.visualization.worldBadgePosition, [0.10 0.18 0.30], [1 1 1]);
apply_axes_theme(axWorld, 'plot');

if strcmp(modeLabel, 'fixed_camera')
    [img, detection] = local_prepare_fixed_camera_display(img, detection);
end

cla(axCam);
imshow(img, 'Parent', axCam);
hold(axCam, 'on');
title(axCam, sprintf('Synthetic camera view (%s)', stageLabel));
if detectionVisible && detection.success
    rectangle(axCam, 'Position', detection.bbox, 'EdgeColor', 'g', 'LineWidth', 1.5);
    plot(axCam, detection.centroid(1), detection.centroid(2), 'go', 'MarkerSize', 8, 'LineWidth', 1.5);
end
badgeColor = [0.30 0.43 0.60];
if strcmp(stageLabel, 'tracking')
    badgeColor = [0.15 0.58 0.28];
elseif strcmp(stageLabel, 'detected')
    badgeColor = [0.85 0.55 0.10];
end
add_status_badge(axCam, upper(stageLabel), cfg.visualization.imageBadgePosition, badgeColor, [1 1 1]);
apply_axes_theme(axCam, 'image');
hold(axCam, 'off');
end

function [displayImg, displayDetection] = local_prepare_fixed_camera_display(img, detection)
displayImg = rot90(img, 2);
displayDetection = detection;
if ~isfield(detection, 'success') || ~detection.success || isempty(img)
    return;
end

[height, width, ~] = size(img);
if isfield(displayDetection, 'centroid') && numel(displayDetection.centroid) == 2 && all(isfinite(displayDetection.centroid))
    displayDetection.centroid = local_rotate_point_180(displayDetection.centroid, width, height);
end
if isfield(displayDetection, 'bbox') && ~isempty(displayDetection.bbox)
    bbox = displayDetection.bbox;
    corners = [ ...
        bbox(1), bbox(2);
        bbox(1) + bbox(3), bbox(2);
        bbox(1) + bbox(3), bbox(2) + bbox(4);
        bbox(1), bbox(2) + bbox(4)];
    corners = local_rotate_points_180(corners, width, height);
    displayDetection.bbox = local_axis_aligned_bbox(corners);
end
if isfield(displayDetection, 'points') && ~isempty(displayDetection.points)
    displayDetection.points = local_rotate_points_180(displayDetection.points, width, height);
end
end

function pointsOut = local_rotate_points_180(pointsIn, width, height)
pointsOut = [width - pointsIn(:, 1) + 1, height - pointsIn(:, 2) + 1];
end

function pointOut = local_rotate_point_180(pointIn, width, height)
pointOut = [width - pointIn(1) + 1, height - pointIn(2) + 1];
end

function bbox = local_axis_aligned_bbox(points)
xMin = min(points(:, 1));
yMin = min(points(:, 2));
xMax = max(points(:, 1));
yMax = max(points(:, 2));
bbox = [xMin, yMin, xMax - xMin, yMax - yMin];
end

function out = local_visibility(flag)
if flag
    out = 'on';
else
    out = 'off';
end
end

function figurePaths = local_save_t2_figures(cfg, figureVisible, outDir, modeLabel, time, targetHistory, estimateHistory, eeHistory, qHistory, errorHistory, detectionArea, detectionVisibleHistory, motionHistory, detectionStartIdx, motionStartIdx)
figurePaths = struct('trajectoryPath', '', 'errorPath', '', 'jointPath', '', 'supportPath', '');
modeTitle = char(modeLabel);
displayLabel = strrep(modeTitle, '_', ' ');
numSamples = numel(time);

trajectoryFig = figure('Name', sprintf('Simulation - T2 %s trajectory', displayLabel), 'Color', 'w', 'Visible', figureVisible, 'Position', [100 100 980 1120]);
tiledlayout(trajectoryFig, 3, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

ax = nexttile;
hold(ax, 'on');
if motionStartIdx > 1
    hTargetStatic = plot(ax, targetHistory(1:motionStartIdx - 1, 1), targetHistory(1:motionStartIdx - 1, 2), 'k-', 'LineWidth', 1.5);
else
    hTargetStatic = plot(ax, targetHistory(:, 1), targetHistory(:, 2), 'k-', 'LineWidth', 1.5);
end
if motionStartIdx <= numSamples
    hTargetMoving = plot(ax, targetHistory(motionStartIdx:end, 1), targetHistory(motionStartIdx:end, 2), '-', 'Color', [0.15 0.65 0.15], 'LineWidth', 1.8);
else
    hTargetMoving = plot(ax, nan, nan, '-', 'Color', [0.15 0.65 0.15], 'LineWidth', 1.8);
end
hEstimate = plot(ax, estimateHistory(:, 1), estimateHistory(:, 2), 'r--', 'LineWidth', 1.2);
hEE = plot(ax, eeHistory(:, 1), eeHistory(:, 2), 'b-.', 'LineWidth', 1.2);
scatter(ax, targetHistory(1, 1), targetHistory(1, 2), 60, [0.18 0.18 0.18], 'filled');
scatter(ax, targetHistory(end, 1), targetHistory(end, 2), 70, 'k', 'filled');
scatter(ax, eeHistory(end, 1), eeHistory(end, 2), 70, 'b', 'filled');
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
title(ax, sprintf('%s path in table frame', displayLabel));
legend(ax, [hTargetStatic, hTargetMoving, hEstimate, hEE], {'Target before motion', 'Target after motion', 'Estimated target', 'End-effector'}, 'Location', 'best');
axis(ax, [cfg.scene.tableX, cfg.scene.tableY]);
axis(ax, 'equal');
grid(ax, 'on');
apply_axes_theme(ax, 'plot');

ax = nexttile;
hold(ax, 'on');
plot(ax, time, targetHistory(:, 1), 'k-', 'LineWidth', 1.3);
plot(ax, time, estimateHistory(:, 1), 'r--', 'LineWidth', 1.1);
plot(ax, time, eeHistory(:, 1), 'b-.', 'LineWidth', 1.1);
if detectionStartIdx <= numSamples
    xline(ax, time(detectionStartIdx), '--b', 'Detection start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
if motionStartIdx <= numSamples
    xline(ax, time(motionStartIdx), '--g', 'Motion start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'X position (m)');
title(ax, 'X position over time');
legend(ax, {'Target', 'Estimated target', 'End-effector'}, 'Location', 'best');
apply_axes_theme(ax, 'plot');

ax = nexttile;
hold(ax, 'on');
plot(ax, time, targetHistory(:, 2), 'k-', 'LineWidth', 1.3);
plot(ax, time, estimateHistory(:, 2), 'r--', 'LineWidth', 1.1);
plot(ax, time, eeHistory(:, 2), 'b-.', 'LineWidth', 1.1);
if detectionStartIdx <= numSamples
    xline(ax, time(detectionStartIdx), '--b', 'Detection start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
if motionStartIdx <= numSamples
    xline(ax, time(motionStartIdx), '--g', 'Motion start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Y position (m)');
title(ax, 'Y position over time');
legend(ax, {'Target', 'Estimated target', 'End-effector'}, 'Location', 'best');
apply_axes_theme(ax, 'plot');
figurePaths.trajectoryPath = fullfile(outDir, sprintf('t2_%s_tracking_trajectory.png', modeTitle));
save_figure(trajectoryFig, figurePaths.trajectoryPath);
close(trajectoryFig);

errorFig = figure('Name', sprintf('Simulation - T2 %s error', displayLabel), 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', errorFig);
plot(ax, time, errorHistory, 'm-', 'LineWidth', 1.5);
if detectionStartIdx <= numSamples
    xline(ax, time(detectionStartIdx), '--b', 'Detection start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
if motionStartIdx <= numSamples
    xline(ax, time(motionStartIdx), '--g', 'Motion start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Position error (m)');
title(ax, sprintf('%s tracking error', displayLabel));
apply_axes_theme(ax, 'plot');
figurePaths.errorPath = fullfile(outDir, sprintf('t2_%s_tracking_error.png', modeTitle));
save_figure(errorFig, figurePaths.errorPath);
close(errorFig);

jointFig = figure('Name', sprintf('Simulation - T2 %s joints', displayLabel), 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', jointFig);
plot(ax, time, qHistory, 'LineWidth', 1.1);
if detectionStartIdx <= numSamples
    xline(ax, time(detectionStartIdx), '--b', 'Detection start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
if motionStartIdx <= numSamples
    xline(ax, time(motionStartIdx), '--g', 'Motion start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Joint angle (rad)');
title(ax, sprintf('%s joint trajectories', displayLabel));
legend(ax, compose('q%d', 1:6), 'Location', 'eastoutside');
apply_axes_theme(ax, 'plot');
figurePaths.jointPath = fullfile(outDir, sprintf('t2_%s_tracking_joints.png', modeTitle));
save_figure(jointFig, figurePaths.jointPath);
close(jointFig);

supportFig = figure('Name', sprintf('Simulation - T2 %s support', displayLabel), 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', supportFig);
yyaxis(ax, 'left');
plot(ax, time, detectionArea, 'g-', 'LineWidth', 1.3);
ylabel(ax, 'Blob area (px)');
yyaxis(ax, 'right');
plot(ax, time, vecnorm(eeHistory(:, 1:2) - targetHistory(:, 1:2), 2, 2), 'b-', 'LineWidth', 1.2);
ylabel(ax, 'EE-target distance (m)');
if detectionStartIdx <= numSamples
    xline(ax, time(detectionStartIdx), '--b', 'Detection start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
if motionStartIdx <= numSamples
    xline(ax, time(motionStartIdx), '--g', 'Motion start', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
end
grid(ax, 'on');
xlabel(ax, 'Time (s)');
title(ax, sprintf('%s detection support and reach distance', displayLabel));
apply_axes_theme(ax, 'plot');
figurePaths.supportPath = fullfile(outDir, sprintf('t2_%s_tracking_support.png', modeTitle));
save_figure(supportFig, figurePaths.supportPath);
close(supportFig);
end
