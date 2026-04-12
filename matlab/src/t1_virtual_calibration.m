function result = t1_virtual_calibration(cfg)
%SIMULATION_T1_VIRTUAL_CALIBRATION Synthetic ChArUco calibration and camera/robot setup.

ensure_dir(cfg.paths.calibration);

cameraTrue = cfg.camera;
board = local_create_charuco_board(cfg);

numViews = cfg.calibration.numViews;
numPoints = size(board.worldPoints2DCalib, 1);
imagePointsArray = nan(numPoints, 2, numViews);
imagePoints = cell(numViews, 1);
cameraPoses = zeros(4, 4, numViews);
cameraPosHistory = zeros(numViews, 3);

theta = linspace(0, 2 * pi, numViews + 1);
theta(end) = [];

for i = 1:numViews
    offset = [0.035 * cos(theta(i)), 0.025 * sin(1.7 * theta(i)), 0.018 * cos(2.2 * theta(i))];
    cameraPos = cfg.calibration.cameraPosition + offset;
    T_wc = lookat_tform(cameraPos, cfg.calibration.lookAtTarget, cfg.calibration.upVector);
    img = local_render_charuco_view(T_wc, cameraTrue, board, cfg.scene.backgroundColor);
    [detectedPoints, ~] = project_points(board.worldPoints3DCalib, T_wc, cameraTrue);
    detectedPoints = detectedPoints + cfg.calibration.noiseStdPx * randn(size(detectedPoints));
    if isempty(detectedPoints) || any(~isfinite(detectedPoints(:))) || size(detectedPoints, 1) ~= numPoints
        error('t1_virtual_calibration:ProjectionFailure', 'Could not project ChArUco corners in view %d.', i);
    end
    imagePoints{i} = detectedPoints;
    imagePointsArray(:, :, i) = detectedPoints;
    cameraPosHistory(i, :) = cameraPos(:).';
    cameraPoses(:, :, i) = T_wc;
end

cameraParams = estimateCameraParameters(imagePointsArray, board.worldPoints2DCalib, ...
    'ImageSize', cameraTrue.imageSize);

[estimatedK, ~, ~, ~, ~] = extract_intrinsics(cameraParams);
trueK = [cameraTrue.fx, 0, cameraTrue.cx;
         0, cameraTrue.fy, cameraTrue.cy;
         0, 0, 1];

perViewError = zeros(numViews, 1);
for i = 1:numViews
    if isprop(cameraParams, 'RotationMatrices') && isprop(cameraParams, 'TranslationVectors')
        uvHat = worldToImage(cameraParams, cameraParams.RotationMatrices(:, :, i), ...
            cameraParams.TranslationVectors(i, :), board.worldPoints3DCalib);
        perViewError(i) = mean(vecnorm(uvHat - imagePoints{i}, 2, 2));
    else
        perViewError(i) = NaN;
    end
end

result = struct();
result.metrics = struct();
result.metrics.trueIntrinsics = trueK;
result.metrics.estimatedIntrinsics = estimatedK;
result.metrics.meanReprojectionError = mean(perViewError, 'omitnan');
result.metrics.perViewError = perViewError;
result.metrics.numViewsUsed = numViews;
result.metrics.cameraPose = cameraPoses(:, :, 1);
result.metrics.boardType = 'charuco-board';

result.history = struct();
result.history.worldPoints2D = board.worldPoints2DCalib;
result.history.worldPoints3D = board.worldPoints3DCalib;
result.history.imagePoints = imagePoints;
result.history.cameraPoses = cameraPoses;
result.history.cameraPositions = cameraPosHistory;

figureVisible = ternary(cfg.showFigures, 'on', 'off');
fig = figure('Name', 'Simulation - T1 ChArUco Calibration', 'Color', 'w', 'Visible', figureVisible);
tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
hold(ax1, 'on');
axis(ax1, 'equal');
title(ax1, 'ChArUco board and camera path');
xlabel(ax1, 'X (m)');
ylabel(ax1, 'Y (m)');
zlabel(ax1, 'Z (m)');
patch(ax1, 'XData', board.boardCornersWorld(:, 1), 'YData', board.boardCornersWorld(:, 2), 'ZData', board.boardCornersWorld(:, 3), ...
    'FaceColor', [0.92 0.92 0.97], 'FaceAlpha', 0.65, 'EdgeColor', [0.35 0.35 0.5], 'LineWidth', 1.2);
plot3(ax1, board.worldPoints3DCalib(:, 1), board.worldPoints3DCalib(:, 2), board.worldPoints3DCalib(:, 3), 'k.', 'MarkerSize', 9);
plot3(ax1, cameraPosHistory(:, 1), cameraPosHistory(:, 2), cameraPosHistory(:, 3), '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
for i = 1:numViews
    plot3(ax1, [cameraPosHistory(i, 1), cfg.calibration.lookAtTarget(1)], ...
        [cameraPosHistory(i, 2), cfg.calibration.lookAtTarget(2)], ...
        [cameraPosHistory(i, 3), cfg.calibration.lookAtTarget(3)], ...
        '-', 'Color', [0.55 0.55 0.55]);
end
draw_camera_frame(ax1, cameraPoses(:, :, 1), 0.035, 'start', 1.6);
draw_camera_frame(ax1, cameraPoses(:, :, end), 0.035, 'end', 1.6);
apply_world_view(ax1, cfg, cfg.scene);
legend(ax1, {'Board plane', 'ChArUco corners', 'Camera path'}, 'Location', 'best');
axis(ax1, [cfg.scene.tableX, cfg.scene.tableY, -0.02, 0.85]);
apply_axes_theme(ax1, 'plot');

ax2 = nexttile;
img = local_render_charuco_view(cameraPoses(:, :, 1), cameraTrue, board, cfg.scene.backgroundColor);
imshow(img, 'Parent', ax2);
title(ax2, 'Synthetic ChArUco image');
if cfg.showFigures
    hold(ax2, 'on');
    plot(ax2, imagePoints{1}(:, 1), imagePoints{1}(:, 2), 'g.', 'MarkerSize', 10);
end
add_status_badge(ax2, 'SYNTHETIC CHArUCO', [0.02 0.86], [0.16 0.18 0.26], [1 1 1]);
apply_axes_theme(ax2, 'image');

ax3 = nexttile([1 2]);
bar(ax3, [trueK(1,1), trueK(2,2), trueK(1,3), trueK(2,3); ...
          estimatedK(1,1), estimatedK(2,2), estimatedK(1,3), estimatedK(2,3)].');
set(ax3, 'XTickLabel', {'fx', 'fy', 'cx', 'cy'});
legend(ax3, {'True', 'Estimated'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
title(ax3, 'True vs estimated intrinsics');
grid(ax3, 'on');
apply_axes_theme(ax3, 'plot');
local_label_bar_values(ax3);

videoPath = '';
if cfg.saveVideos
    videoPath = fullfile(cfg.paths.calibration, 't1_calibration_animation.mp4');
    [writer, videoPath] = open_video_writer(videoPath, cfg.videoFrameRate);
    cleanup = onCleanup(@() close(writer)); %#ok<NASGU>
    for i = 1:numViews
        cla(ax1);
        hold(ax1, 'on');
        grid(ax1, 'on');
        axis(ax1, 'equal');
        xlabel(ax1, 'X (m)');
        ylabel(ax1, 'Y (m)');
        zlabel(ax1, 'Z (m)');
        title(ax1, sprintf('ChArUco calibration view %d / %d', i, numViews));
        patch(ax1, 'XData', board.boardCornersWorld(:, 1), 'YData', board.boardCornersWorld(:, 2), 'ZData', board.boardCornersWorld(:, 3), ...
            'FaceColor', [0.92 0.92 0.97], 'FaceAlpha', 0.65, 'EdgeColor', [0.35 0.35 0.5], 'LineWidth', 1.2);
        plot3(ax1, board.worldPoints3DCalib(:, 1), board.worldPoints3DCalib(:, 2), board.worldPoints3DCalib(:, 3), 'k.', 'MarkerSize', 9);
        plot3(ax1, cameraPosHistory(1:i, 1), cameraPosHistory(1:i, 2), cameraPosHistory(1:i, 3), '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
        plot3(ax1, cameraPosHistory(i, 1), cameraPosHistory(i, 2), cameraPosHistory(i, 3), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
        plot3(ax1, [cameraPosHistory(i, 1), cfg.calibration.lookAtTarget(1)], ...
            [cameraPosHistory(i, 2), cfg.calibration.lookAtTarget(2)], ...
            [cameraPosHistory(i, 3), cfg.calibration.lookAtTarget(3)], ...
            '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2);
        draw_camera_frame(ax1, cameraPoses(:, :, i), 0.035, sprintf('%d/%d', i, numViews), 1.5);
        apply_world_view(ax1, cfg, cfg.scene);
        axis(ax1, [cfg.scene.tableX, cfg.scene.tableY, -0.02, 0.85]);

        img = local_render_charuco_view(cameraPoses(:, :, i), cameraTrue, board, cfg.scene.backgroundColor);
        imshow(img, 'Parent', ax2);
        if cfg.showFigures
            hold(ax2, 'on');
            plot(ax2, imagePoints{i}(:, 1), imagePoints{i}(:, 2), 'g.', 'MarkerSize', 10);
        end
        add_status_badge(ax2, sprintf('VIEW %d / %d', i, numViews), [0.02 0.86], [0.16 0.18 0.26], [1 1 1]);
        title(ax2, sprintf('Synthetic ChArUco image %d', i));
        drawnow;
        frame = getframe(fig);
        writeVideo(writer, frame);
        if cfg.showFigures
            hold(ax2, 'off');
        end
    end
end

close(fig);

figurePaths = struct('boardPath', '', 'imagePath', '', 'intrinsicsPath', '');
if cfg.saveFigures
    figurePaths = local_save_t1_figures(cfg, figureVisible, board, cameraPosHistory, cameraPoses, imagePoints, trueK, estimatedK, cameraTrue);
end

result.paths = struct();
result.paths.printableBoardPath = board.texturePath;
result.paths.figure = figurePaths.boardPath;
result.paths.figures = figurePaths;
result.paths.video = videoPath;
result.cameraParams = cameraParams;
end

function board = local_create_charuco_board(cfg)
charuco = cfg.calibration.charuco;
board = struct();
board.patternDims = charuco.patternDims;
board.markerFamily = charuco.markerFamily;
board.checkerSize = charuco.checkerSize;
board.markerSize = charuco.markerSize;
board.texturePath = fullfile(cfg.paths.calibration, 't1_calibration_charuco_board_printable.png');

board.texture = generateCharucoBoard(charuco.textureImageSize, board.patternDims, board.markerFamily, ...
    charuco.textureCheckerSizePx, charuco.textureMarkerSizePx, ...
    MarginSize=1, ...
    MinMarkerID=charuco.minMarkerID, ...
    OriginCheckerColor=charuco.originCheckerColor);
imwrite(board.texture, board.texturePath);
board.texture = imread(board.texturePath);
board.textureRgb = repmat(board.texture, 1, 1, 3);

rawPoints2D = double(patternWorldPoints("charuco-board", board.patternDims, board.checkerSize));
board.worldPoints2D = rawPoints2D;
board.worldPoints2DCalib = rawPoints2D - mean(rawPoints2D, 1) + cfg.calibration.boardCenter(1:2);
board.worldPoints3DCalib = [board.worldPoints2DCalib, cfg.calibration.boardCenter(3) * ones(size(board.worldPoints2DCalib, 1), 1)];

board.boardSize = [board.patternDims(2) * board.checkerSize, board.patternDims(1) * board.checkerSize];
board.boardOriginWorld = [cfg.calibration.boardCenter(1) - board.boardSize(1) / 2, ...
                          cfg.calibration.boardCenter(2) - board.boardSize(2) / 2, ...
                          cfg.calibration.boardCenter(3)];
board.boardCornersWorld = [board.boardOriginWorld;
                           board.boardOriginWorld + [board.boardSize(1), 0, 0];
                           board.boardOriginWorld + [board.boardSize(1), board.boardSize(2), 0];
                           board.boardOriginWorld + [0, board.boardSize(2), 0]];
end

function img = local_render_charuco_view(T_wc, cameraTrue, board, backgroundColor)
imageSize = cameraTrue.imageSize;
img = zeros(imageSize(1), imageSize(2), 3, 'uint8');
for c = 1:3
    img(:, :, c) = backgroundColor(c);
end

[uv, depth] = project_points(board.boardCornersWorld, T_wc, cameraTrue);
if any(~isfinite(uv(:))) || any(depth <= 0)
    return;
end

sourceCorners = [1, 1;
                 size(board.texture, 2), 1;
                 size(board.texture, 2), size(board.texture, 1);
                 1, size(board.texture, 1)];
outputRef = imref2d(imageSize);
tform = fitgeotrans(sourceCorners, uv, 'projective');
warpedRgb = imwarp(board.textureRgb, tform, 'OutputView', outputRef);
warpedMask = imwarp(true(size(board.texture, 1), size(board.texture, 2)), tform, 'OutputView', outputRef);

for c = 1:3
    channel = img(:, :, c);
    warpedChannel = warpedRgb(:, :, c);
    channel(warpedMask) = warpedChannel(warpedMask);
    img(:, :, c) = channel;
end
end

function y = ternary(cond, a, b)
if cond
    y = a;
else
    y = b;
end
end

function figurePaths = local_save_t1_figures(cfg, figureVisible, board, cameraPosHistory, cameraPoses, imagePoints, trueK, estimatedK, cameraTrue)
figurePaths = struct('boardPath', '', 'imagePath', '', 'intrinsicsPath', '');

boardFig = figure('Name', 'Simulation - T1 board path', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', boardFig);
hold(ax, 'on');
axis(ax, 'equal');
title(ax, 'ChArUco board and camera path');
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
patch(ax, 'XData', board.boardCornersWorld(:, 1), 'YData', board.boardCornersWorld(:, 2), 'ZData', board.boardCornersWorld(:, 3), ...
    'FaceColor', [0.92 0.92 0.97], 'FaceAlpha', 0.65, 'EdgeColor', [0.35 0.35 0.5], 'LineWidth', 1.2);
plot3(ax, board.worldPoints3DCalib(:, 1), board.worldPoints3DCalib(:, 2), board.worldPoints3DCalib(:, 3), 'k.', 'MarkerSize', 9);
plot3(ax, cameraPosHistory(:, 1), cameraPosHistory(:, 2), cameraPosHistory(:, 3), '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
for i = 1:size(cameraPosHistory, 1)
    plot3(ax, [cameraPosHistory(i, 1), cfg.calibration.lookAtTarget(1)], ...
        [cameraPosHistory(i, 2), cfg.calibration.lookAtTarget(2)], ...
        [cameraPosHistory(i, 3), cfg.calibration.lookAtTarget(3)], ...
        '-', 'Color', [0.55 0.55 0.55]);
end
draw_camera_frame(ax, cameraPoses(:, :, 1), 0.035, 'start', 1.6);
draw_camera_frame(ax, cameraPoses(:, :, end), 0.035, 'end', 1.6);
apply_world_view(ax, cfg, cfg.scene);
legend(ax, {'Board plane', 'ChArUco corners', 'Camera path'}, 'Location', 'best');
axis(ax, [cfg.scene.tableX, cfg.scene.tableY, -0.02, 0.85]);
apply_axes_theme(ax, 'plot');
figurePaths.boardPath = fullfile(cfg.paths.calibration, 't1_calibration_board_path.png');
save_figure(boardFig, figurePaths.boardPath);
close(boardFig);

imageFig = figure('Name', 'Simulation - T1 synthetic image', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', imageFig);
img = local_render_charuco_view(cameraPoses(:, :, 1), cameraTrue, board, cfg.scene.backgroundColor);
imshow(img, 'Parent', ax);
hold(ax, 'on');
plot(ax, imagePoints{1}(:, 1), imagePoints{1}(:, 2), 'g.', 'MarkerSize', 10);
title(ax, 'Synthetic ChArUco image using printable board texture');
add_status_badge(ax, 'SYNTHETIC CHArUCO', [0.02 0.86], [0.16 0.18 0.26], [1 1 1]);
apply_axes_theme(ax, 'image');
figurePaths.imagePath = fullfile(cfg.paths.calibration, 't1_calibration_synthetic_image.png');
save_figure(imageFig, figurePaths.imagePath);
close(imageFig);

intrinsicsFig = figure('Name', 'Simulation - T1 intrinsics', 'Color', 'w', 'Visible', figureVisible);
ax = axes('Parent', intrinsicsFig);
bar(ax, [trueK(1,1), trueK(2,2), trueK(1,3), trueK(2,3); ...
          estimatedK(1,1), estimatedK(2,2), estimatedK(1,3), estimatedK(2,3)].');
set(ax, 'XTickLabel', {'fx', 'fy', 'cx', 'cy'});
legend(ax, {'True', 'Estimated'}, 'Location', 'northoutside', 'Orientation', 'horizontal');
title(ax, 'True vs estimated intrinsics');
apply_axes_theme(ax, 'plot');
local_label_bar_values(ax);
figurePaths.intrinsicsPath = fullfile(cfg.paths.calibration, 't1_calibration_intrinsics.png');
save_figure(intrinsicsFig, figurePaths.intrinsicsPath);
close(intrinsicsFig);
end

function local_label_bar_values(ax)
bars = findobj(ax, 'Type', 'Bar');
if isempty(bars)
    return;
end
for i = 1:numel(bars)
    values = bars(i).YData;
    x = bars(i).XEndPoints;
    for j = 1:numel(values)
        text(ax, x(j), values(j), sprintf('  %.0f', values(j)), ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 9, ...
            'FontWeight', 'bold', ...
            'Color', [0.2 0.2 0.2]);
    end
end
end
