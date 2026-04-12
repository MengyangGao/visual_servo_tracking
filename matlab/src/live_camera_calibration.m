function results = live_camera_calibration(cfg)
%LIVE_CAMERA_CALIBRATION Capture ChArUco images from a real webcam and calibrate.

if nargin < 1 || isempty(cfg)
    cfg = config();
end

ensure_dir(cfg.paths.liveCalibration);
rawFramesDir = fullfile(cfg.paths.liveCalibration, 'raw_frames');
ensure_dir(rawFramesDir);

logPath = fullfile(cfg.paths.liveCalibration, 'live_camera_log.txt');
if isfile(logPath)
    delete(logPath);
end
diary(logPath);
logCleanup = onCleanup(@() diary('off')); %#ok<NASGU>

board = charuco_board_asset(cfg);
fprintf('[LiveCamera] Printable board PNG: %s\n', board.paths.png);
fprintf('[LiveCamera] Printable board PDF: %s\n', board.paths.pdf);
fprintf('[LiveCamera] Print the board at 100%% scale and keep it flat.\n');
pre_capture_pause(cfg.realCamera.preCaptureDelaySec, 'LiveCamera');

cameraName = local_select_camera(cfg.realCamera.cameraNamePreference);
cam = webcam(cameraName);
camCleanup = onCleanup(@() local_release_camera(cam)); %#ok<NASGU>

fprintf('[LiveCamera] Using camera: %s\n', cameraName);
if isprop(cam, 'AvailableResolutions')
    fprintf('[LiveCamera] Available resolutions: %s\n', strjoin(string(cam.AvailableResolutions), ', '));
end

previewFig = [];
previewAx = [];
if cfg.realCamera.preview && cfg.showFigures
    previewFig = figure('Name', 'Live ChArUco Capture', 'Color', 'w');
    previewAx = axes('Parent', previewFig);
    title(previewAx, 'Live camera preview');
end

acceptedImages = {};
acceptedPoints = {};
acceptedFrameIds = [];
attempts = 0;
maxAttempts = max(cfg.realCamera.captureCount * 4, cfg.realCamera.captureCount + 6);

while numel(acceptedImages) < cfg.realCamera.captureCount && attempts < maxAttempts
    attempts = attempts + 1;
    frame = snapshot(cam);

    if cfg.realCamera.saveRawFrames
        imwrite(frame, fullfile(rawFramesDir, sprintf('frame_%03d.png', attempts)));
    end

    [pts, used] = detectCharucoBoardPoints( ...
        frame, ...
        board.patternDims, ...
        board.markerFamily, ...
        board.checkerSize, ...
        board.markerSize, ...
        MinMarkerID=board.minMarkerID, ...
        OriginCheckerColor=board.originCheckerColor, ...
        RefineCorners=true);

    accepted = used && ~isempty(pts) && all(isfinite(pts(:))) && size(pts, 1) == size(board.worldPoints, 1);
    if accepted
        acceptedImages{end + 1} = frame; %#ok<AGROW>
        acceptedPoints{end + 1} = pts; %#ok<AGROW>
        acceptedFrameIds(end + 1) = attempts; %#ok<AGROW>
        fprintf('[LiveCamera] Accepted frame %d (%d / %d)\n', attempts, numel(acceptedImages), cfg.realCamera.captureCount);
    else
        fprintf('[LiveCamera] Rejected frame %d\n', attempts);
    end

    if ~isempty(previewFig) && isvalid(previewFig)
        imshow(frame, 'Parent', previewAx);
        hold(previewAx, 'on');
        if accepted
            plot(previewAx, pts(:, 1), pts(:, 2), 'g.', 'MarkerSize', 10);
        end
        hold(previewAx, 'off');
        title(previewAx, sprintf('Frame %d | accepted %d / %d', attempts, numel(acceptedImages), cfg.realCamera.captureCount));
        drawnow;
    end

    pause(cfg.realCamera.captureIntervalSec);
end

if numel(acceptedImages) < cfg.realCamera.requiredDetections
    error('live_camera_calibration:InsufficientViews', ...
        'Only %d valid ChArUco frames were captured; required at least %d.', ...
        numel(acceptedImages), cfg.realCamera.requiredDetections);
end

imageSize = [size(acceptedImages{1}, 1), size(acceptedImages{1}, 2)];
calib = calibrate_charuco_images(acceptedImages, board, imageSize, cfg.realCamera.requiredDetections);

cameraParams = calib.cameraParams;
[estimatedK, distortionCoeffs] = local_extract_intrinsics(cameraParams);

summaryPath = fullfile(cfg.paths.liveCalibration, 'live_camera_intrinsics.png');
captureGridPath = fullfile(cfg.paths.liveCalibration, 'live_camera_captures.png');
supportPath = fullfile(cfg.paths.liveCalibration, 'live_camera_support.png');

figureVisible = ternary(cfg.showFigures, 'on', 'off');
local_save_capture_grid(captureGridPath, figureVisible, acceptedImages, acceptedPoints, cfg.realCamera.captureCount);
local_save_intrinsics_figure(summaryPath, figureVisible, estimatedK, distortionCoeffs, calib.metrics);
local_save_support_figure(supportPath, figureVisible, acceptedFrameIds, calib.metrics);

cameraParamsPath = fullfile(cfg.paths.liveCalibration, 'camera_params.mat');
boardSpec = rmfield(board, 'image');
save(cameraParamsPath, 'cameraParams', 'boardSpec', 'calib');

results = struct();
results.board = board;
results.cameraParams = cameraParams;
results.metrics = calib.metrics;
results.paths = struct();
results.paths.logPath = logPath;
results.paths.cameraParamsPath = cameraParamsPath;
results.paths.printableBoardPng = board.paths.png;
results.paths.printableBoardPdf = board.paths.pdf;
results.paths.captureGrid = captureGridPath;
results.paths.intrinsicsFigure = summaryPath;
results.paths.supportFigure = supportPath;
results.paths.rawFramesDir = rawFramesDir;
results.paths.acceptedFrameIds = acceptedFrameIds;

fprintf('[LiveCamera] Mean reprojection error: %.4f px\n', results.metrics.meanReprojectionError);
fprintf('[LiveCamera] Estimated fx/fy: %.2f / %.2f\n', estimatedK(1,1), estimatedK(2,2));
fprintf('[LiveCamera] Log stored at %s\n', logPath);
end

function cameraName = local_select_camera(preferenceList)
available = string(webcamlist);
if isempty(available)
    error('live_camera_calibration:NoCamera', 'No webcam devices were detected.');
end

if nargin < 1 || isempty(preferenceList)
    cameraName = available(1);
    return;
end

preferences = string(preferenceList);
for i = 1:numel(preferences)
    idx = find(strcmpi(available, preferences(i)), 1);
    if ~isempty(idx)
        cameraName = available(idx);
        return;
    end
end

cameraName = available(1);
end

function local_release_camera(cam)
try
    delete(cam);
catch
end
end

function [K, distortionCoeffs] = local_extract_intrinsics(cameraParams)
if isprop(cameraParams, 'IntrinsicMatrix')
    K = cameraParams.IntrinsicMatrix.';
    distortionCoeffs = cameraParams.RadialDistortion;
    if isprop(cameraParams, 'TangentialDistortion')
        distortionCoeffs = [distortionCoeffs, cameraParams.TangentialDistortion];
    end
else
    K = cameraParams.Intrinsics.K;
    distortionCoeffs = cameraParams.Intrinsics.RadialDistortion;
    if isprop(cameraParams.Intrinsics, 'TangentialDistortion')
        distortionCoeffs = [distortionCoeffs, cameraParams.Intrinsics.TangentialDistortion];
    end
end
end

function local_save_capture_grid(outPath, figureVisible, acceptedImages, acceptedPoints, desiredCount)
fig = figure('Name', 'Live ChArUco Captures', 'Color', 'w', 'Visible', figureVisible);
numShow = min(numel(acceptedImages), 6);
tiledlayout(fig, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
for i = 1:numShow
    ax = nexttile;
    imshow(acceptedImages{i}, 'Parent', ax);
    hold(ax, 'on');
    plot(ax, acceptedPoints{i}(:, 1), acceptedPoints{i}(:, 2), 'g.', 'MarkerSize', 8);
    title(ax, sprintf('Accepted %d', i));
    apply_axes_theme(ax, 'image');
    hold(ax, 'off');
end
for i = numShow + 1:6
    ax = nexttile;
    axis(ax, 'off');
    text(ax, 0.5, 0.5, sprintf('%d / %d', numel(acceptedImages), desiredCount), ...
        'HorizontalAlignment', 'center', 'FontSize', 16);
end
save_figure(fig, outPath);
close(fig);
end

function local_save_intrinsics_figure(outPath, figureVisible, estimatedK, distortionCoeffs, metrics)
fig = figure('Name', 'Live ChArUco Intrinsics', 'Color', 'w', 'Visible', figureVisible);
tiledlayout(fig, 2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
bar(ax1, [estimatedK(1,1), estimatedK(2,2), estimatedK(1,3), estimatedK(2,3)]);
set(ax1, 'XTickLabel', {'fx', 'fy', 'cx', 'cy'});
title(ax1, 'Estimated intrinsics');
grid(ax1, 'on');
apply_axes_theme(ax1, 'plot');

ax2 = nexttile;
bar(ax2, [metrics.meanReprojectionError, mean(metrics.perViewError), max(metrics.perViewError)]);
set(ax2, 'XTickLabel', {'mean reproj', 'mean per-view', 'max per-view'});
title(ax2, sprintf('Distortion coeffs: %s', mat2str(distortionCoeffs, 4)));
grid(ax2, 'on');
apply_axes_theme(ax2, 'plot');

save_figure(fig, outPath);
close(fig);
end

function local_save_support_figure(outPath, figureVisible, acceptedFrameIds, metrics)
fig = figure('Name', 'Live ChArUco Support', 'Color', 'w', 'Visible', figureVisible);
yyaxis left;
plot(acceptedFrameIds, metrics.perViewError, '-o', 'LineWidth', 1.2);
ylabel('Reprojection error (px)');
yyaxis right;
plot(acceptedFrameIds, 1:numel(acceptedFrameIds), '-s', 'LineWidth', 1.0);
ylabel('Accepted frame index');
xlabel('Capture attempt');
title('Capture progression and reprojection error');
grid on;
apply_axes_theme(ax, 'plot');
save_figure(fig, outPath);
close(fig);
end

function y = ternary(cond, a, b)
if cond
    y = a;
else
    y = b;
end
end
