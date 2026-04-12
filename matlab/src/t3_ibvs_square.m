function result = t3_ibvs_square(cfg, runMode)
%SIMULATION_T3_IBVS_SQUARE Feature-based visual servoing for a square target.

ensure_dir(cfg.paths.ibvs);

if nargin < 2 || isempty(runMode)
    runMode = 'auto';
end
runMode = normalize_run_mode(runMode);
isManual = strcmp(runMode, "manual");

robot = cfg.robot;
eeName = cfg.eeName;
ik = inverseKinematics('RigidBodyTree', robot);
weights = [0.25 0.25 0.25 1 1 1];

squareCorners = square_corners(cfg.t3.squareCenter, cfg.t3.squareSide);
scene = struct();
scene.backgroundColor = uint8([248 248 248]);
scene.objects = struct('kind', 'square', 'name', 'square_target', 'corners', squareCorners, ...
    'fillColor', uint8([235 235 245]), 'edgeColor', uint8([20 20 20]), ...
    'cornerColor', uint8([30 30 30]), 'markCorners', true);

desiredT = local_downward_pose(cfg.t3.desiredCameraPosition);
[sDesired, desiredPixels, desiredDepth] = local_feature_vector(desiredT, cfg.camera, squareCorners);
if any(~isfinite(sDesired))
    error('t3_ibvs_square:DesiredFeaturesInvalid', 'Desired features could not be projected.');
end

initialT = local_downward_pose(cfg.t3.initialCameraPosition);
[q, solInfo] = ik(eeName, initialT, weights, zeros(1, 6));
if ~isstruct(solInfo) || ~isfield(solInfo, 'ExitFlag') || solInfo.ExitFlag <= 0
    error('t3_ibvs_square:InitialIKFailure', 'Unable to find an initial camera pose for IBVS.');
end
q = local_clamp_to_limits(robot, q);

stepLimit = cfg.t3.numSteps;
if isManual
    stepLimit = Inf;
end

timeHistory = zeros(0, 1);
qHistory = zeros(0, 6);
featureHistory = zeros(0, numel(sDesired));
errorHistory = zeros(0, 1);
depthHistory = zeros(0, 4);
twistHistory = zeros(0, 6);
eeHistory = zeros(0, 3);

figureVisible = local_visibility(cfg.showFigures);
fig = figure('Name', 'Simulation - T3 IBVS square tracking', 'Color', 'w', 'Visible', figureVisible);
tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
axWorld = nexttile;
axCam = nexttile;

if strcmp(runMode, "manual") && ~cfg.showFigures
    error('t3_ibvs_square:ManualModeRequiresVisibleFigure', 'Manual mode requires visible figures.');
end
session = session_controls(fig, runMode, 'T3 IBVS square');
if session.manual
    session.waitForStart();
    if session.shouldStop()
        close(fig);
        error('t3_ibvs_square:StoppedBeforeStart', 'The session was stopped before tracking started.');
    end
end

if cfg.saveVideos
    videoFile = fullfile(cfg.paths.ibvs, 't3_ibvs_square_tracking.mp4');
    [writer, videoPath] = open_video_writer(videoFile, cfg.videoFrameRate);
    cleanup = onCleanup(@() close(writer));
else
    videoPath = '';
end

k = 0;
while true
    if session.shouldStop()
        break;
    end
    if isfinite(stepLimit) && k >= stepLimit
        break;
    end

    k = k + 1;
    currentTime = (k - 1) * cfg.t3.dt;
    timeHistory(k, 1) = currentTime;

    T_wc = getTransform(robot, q, eeName);
    [s, pixels, depths] = local_feature_vector(T_wc, cfg.camera, squareCorners);
    if any(~isfinite(s))
        error('t3_ibvs_square:ProjectionFailure', 'Square corners are not visible at step %d.', k);
    end

    e = s - sDesired;
    L = local_translation_matrix(s, depths);
    vCam = -cfg.t3.ibvsGain * dls_solve(L, e, cfg.t3.interactionDamping);
    [q, sNext, pixelsNext, depthsNext, nextError] = local_backtracking_translation_step( ...
        robot, q, T_wc, vCam, cfg.t3.dt, eeName, cfg.camera, squareCorners, sDesired, weights, ik);

    qHistory(k, :) = q;
    featureHistory(k, :) = sNext.';
    errorHistory(k, 1) = nextError;
    depthHistory(k, :) = depthsNext.';
    twistHistory(k, :) = [vCam(:).', 0, 0, 0];
    T_now = getTransform(robot, q, eeName);
    eeHistory(k, :) = T_now(1:3, 4).';

    local_draw_t3_frame(axWorld, axCam, robot, q, eeName, cfg, scene, pixelsNext, sDesired, desiredPixels, sNext - sDesired, k, stepLimit);
    drawnow;
    if cfg.saveVideos
        frame = getframe(fig);
        writeVideo(writer, frame);
    end
end

time = timeHistory;
actualSteps = numel(time);
if actualSteps == 0
    error('t3_ibvs_square:NoSamples', 'The T3 session produced no samples.');
end

session.cleanup();
if isgraphics(fig)
    delete(fig);
end

figurePaths = struct('errorPath', '', 'featurePath', '', 'jointPath', '', 'depthPath', '');
if cfg.saveFigures
    figurePaths = local_save_t3_figures(cfg, figureVisible, time, errorHistory, featureHistory, qHistory, depthHistory, sDesired);
end

result = struct();
result.runMode = char(runMode);
result.metrics = struct();
result.metrics.finalFeatureError = errorHistory(end);
result.metrics.rmseFeatureError = sqrt(mean(errorHistory .^ 2));
result.metrics.finalCameraPosition = eeHistory(end, :);
result.metrics.desiredCameraPosition = cfg.t3.desiredCameraPosition;
result.metrics.finalDepthMean = mean(depthHistory(end, :));
result.metrics.desiredFeatureNorm = norm(sDesired);

result.history = struct();
result.history.time = time;
result.history.q = qHistory;
result.history.features = featureHistory;
result.history.error = errorHistory;
result.history.depths = depthHistory;
result.history.twist = twistHistory;
result.history.ee = eeHistory;
result.history.desiredFeatures = sDesired;
result.history.desiredPixels = desiredPixels;
result.history.desiredDepth = desiredDepth;

result.paths = struct();
result.paths.figure = figurePaths.errorPath;
result.paths.figures = figurePaths;
result.paths.video = videoPath;
end

function [s, pixels, depths] = local_feature_vector(T_wc, camera, corners)
[pixels, depths] = project_points(corners, T_wc, camera);
if any(~isfinite(pixels(:))) || any(depths <= 0)
    s = nan(numel(pixels), 1);
    return;
end
normCoords = [((pixels(:, 1) - camera.cx) ./ camera.fx), ((pixels(:, 2) - camera.cy) ./ camera.fy)];
s = reshape(normCoords.', [], 1);
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

function local_draw_t3_frame(axWorld, axCam, robot, q, eeName, cfg, scene, pixels, sDesired, desiredPixels, e, stepIdx, totalSteps)
cla(axWorld);
hold(axWorld, 'on');
axis(axWorld, [cfg.scene.tableX, cfg.scene.tableY, -0.02, 0.9]);
axis(axWorld, 'equal');
xlabel(axWorld, 'X (m)');
ylabel(axWorld, 'Y (m)');
zlabel(axWorld, 'Z (m)');
if isfinite(totalSteps)
    title(axWorld, sprintf('T3 IBVS step %d/%d', stepIdx, totalSteps));
else
    title(axWorld, sprintf('T3 IBVS step %d', stepIdx));
end

patch(axWorld, 'XData', [cfg.scene.tableX(1), cfg.scene.tableX(2), cfg.scene.tableX(2), cfg.scene.tableX(1)], ...
    'YData', [cfg.scene.tableY(1), cfg.scene.tableY(1), cfg.scene.tableY(2), cfg.scene.tableY(2)], ...
    'ZData', [cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ, cfg.scene.tableZ], ...
    'FaceColor', [0.95 0.93 0.89], 'FaceAlpha', 0.75, 'EdgeColor', [0.35 0.3 0.25]);

square = scene.objects(1).corners;
plot3(axWorld, [square(:,1); square(1,1)], [square(:,2); square(1,2)], [square(:,3); square(1,3)], ...
    'k-', 'LineWidth', 1.5);
scatter3(axWorld, square(:, 1), square(:, 2), square(:, 3), 120, 'k', 'filled');
T_wc = getTransform(robot, q, eeName);
plot3(axWorld, T_wc(1, 4), T_wc(2, 4), T_wc(3, 4), 'bp', 'MarkerSize', 14, 'MarkerFaceColor', 'b');
show(robot, q, 'Parent', axWorld, 'Frames', 'off', 'PreservePlot', false, 'Visuals', 'on');
draw_camera_frame(axWorld, T_wc, 0.055, 'cam', 1.5);
apply_world_view(axWorld, cfg, cfg.scene);
add_status_badge(axWorld, 'IBVS | TRACKING', cfg.visualization.worldBadgePosition, [0.10 0.18 0.30], [1 1 1]);
apply_axes_theme(axWorld, 'plot');

cla(axCam);
img = local_render_square_image(T_wc, cfg.camera, scene);
imshow(img, 'Parent', axCam);
hold(axCam, 'on');
title(axCam, sprintf('Camera image (feature error = %.4f)', norm(e)));
plot(axCam, desiredPixels(:, 1), desiredPixels(:, 2), 'gx', 'MarkerSize', 10, 'LineWidth', 1.5);
plot(axCam, pixels(:, 1), pixels(:, 2), 'ro', 'MarkerSize', 6, 'LineWidth', 1.5);
for i = 1:size(pixels, 1)
    plot(axCam, [pixels(i, 1), desiredPixels(i, 1)], [pixels(i, 2), desiredPixels(i, 2)], 'y-', 'LineWidth', 0.9);
end
add_status_badge(axCam, 'CURRENT vs DESIRED FEATURES', cfg.visualization.imageBadgePosition, [0.16 0.18 0.26], [1 1 1]);
apply_axes_theme(axCam, 'image');
hold(axCam, 'off');
end

function img = local_render_square_image(T_wc, camera, scene)
[img, ~] = render_camera_view(T_wc, camera, scene);
end

function out = local_visibility(flag)
if flag
    out = 'on';
else
    out = 'off';
end
end

function [qBest, sBest, pixelsBest, depthsBest, errorBest] = local_backtracking_translation_step( ...
    robot, q, T_wc, vCam, dt, eeName, camera, squareCorners, sDesired, weights, ik)

currentT = T_wc;
[sCurrent, pixelsCurrent, depthsCurrent] = local_feature_vector(currentT, camera, squareCorners);
currentError = norm(sCurrent - sDesired);

stepScale = 1.0;
qBest = q;
sBest = sCurrent;
pixelsBest = pixelsCurrent;
depthsBest = depthsCurrent;
errorBest = currentError;

for trial = 1:8
    candidateT = local_apply_body_translation(T_wc, vCam, stepScale * dt);
    [sCand, pixelsCand, depthsCand] = local_feature_vector(candidateT, camera, squareCorners);
    if all(isfinite(sCand))
        errCand = norm(sCand - sDesired);
        if errCand <= currentError
            [candidateQ, solInfo] = ik(eeName, candidateT, weights, q);
            if isstruct(solInfo) && isfield(solInfo, 'ExitFlag') && solInfo.ExitFlag > 0
                candidateQ = local_clamp_to_limits(robot, candidateQ);
                qBest = candidateQ;
                sBest = sCand;
                pixelsBest = pixelsCand;
                depthsBest = depthsCand;
                errorBest = errCand;
                return;
            end
        end
    end
    stepScale = stepScale * 0.5;
end

if ~isfinite(errorBest)
    error('t3_ibvs_square:LineSearchFailure', 'Unable to find a valid IBVS update step.');
end
end

function T_new = local_apply_body_translation(T_current, vCam, dt)
T_new = T_current;
T_new(1:3, 4) = T_current(1:3, 4) + T_current(1:3, 1:3) * (vCam(:) * dt);
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

function figurePaths = local_save_t3_figures(cfg, figureVisible, time, errorHistory, featureHistory, qHistory, depthHistory, sDesired)
figurePaths = struct('errorPath', '', 'featurePath', '', 'jointPath', '', 'depthPath', '');

errorFig = figure('Name', 'Simulation - T3 IBVS error', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', errorFig);
plot(ax, time, errorHistory, 'm-', 'LineWidth', 1.5);
if isfield(cfg.t3, 'featureTolerance')
    yline(ax, cfg.t3.featureTolerance, '--', 'Tolerance', 'Color', [0.15 0.55 0.20], 'LabelVerticalAlignment', 'bottom');
end
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Feature error norm');
title(ax, 'Feature error over time');
apply_axes_theme(ax, 'plot');
figurePaths.errorPath = fullfile(cfg.paths.ibvs, 't3_ibvs_square_tracking_error.png');
save_figure(errorFig, figurePaths.errorPath);
close(errorFig);

featureFig = figure('Name', 'Simulation - T3 IBVS features', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', featureFig);
plot(ax, time, featureHistory, 'LineWidth', 1.1);
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Normalized image feature');
title(ax, 'Normalized image features and desired references');
legend(ax, compose('s%d', 1:numel(sDesired)), 'Location', 'eastoutside');
apply_axes_theme(ax, 'plot');
figurePaths.featurePath = fullfile(cfg.paths.ibvs, 't3_ibvs_square_tracking_features.png');
save_figure(featureFig, figurePaths.featurePath);
close(featureFig);

jointFig = figure('Name', 'Simulation - T3 IBVS joints', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', jointFig);
plot(ax, time, qHistory, 'LineWidth', 1.0);
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Joint angle (rad)');
title(ax, 'Joint motion');
legend(ax, compose('q%d', 1:6), 'Location', 'eastoutside');
apply_axes_theme(ax, 'plot');
figurePaths.jointPath = fullfile(cfg.paths.ibvs, 't3_ibvs_square_tracking_joints.png');
save_figure(jointFig, figurePaths.jointPath);
close(jointFig);

depthFig = figure('Name', 'Simulation - T3 IBVS depths', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', depthFig);
plot(ax, time, depthHistory, 'LineWidth', 1.0);
grid(ax, 'on');
xlabel(ax, 'Time (s)');
ylabel(ax, 'Depth (m)');
title(ax, 'Corner depth evolution');
legend(ax, compose('Z%d', 1:4), 'Location', 'eastoutside');
apply_axes_theme(ax, 'plot');
figurePaths.depthPath = fullfile(cfg.paths.ibvs, 't3_ibvs_square_tracking_depths.png');
save_figure(depthFig, figurePaths.depthPath);
close(depthFig);
end
