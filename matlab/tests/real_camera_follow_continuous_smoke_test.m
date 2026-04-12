function real_camera_follow_continuous_smoke_test()
%REAL_CAMERA_FOLLOW_CONTINUOUS_SMOKE_TEST Smoke test for the legacy follow wrapper.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(rootDir, 'src')));

calibrationDir = fullfile(rootDir, 'results', 't1_live_calibration', 'calibration');
cameraParamsPath = fullfile(calibrationDir, 'cameraParams.mat');
imagePath = fullfile(calibrationDir, 'Image1.png');
if ~isfile(cameraParamsPath) || ~isfile(imagePath)
    fprintf('REAL_CAMERA_FOLLOW_CONTINUOUS_SMOKE_TEST_SKIPPED\n');
    return;
end

result = run_real_camera_follow_continuous( ...
    'showFigures', false, ...
    'saveFigures', false, ...
    'saveVideos', false, ...
    'RunMode', 'auto', ...
    'SourceMode', 'files', ...
    'MaxFrames', 4, ...
    'PreCaptureDelaySec', 0);

assert(strcmp(result.source, 'files'), 'Follow wrapper smoke test should use file sequence.');
assert(result.metrics.samplesProcessed == 4, 'Follow wrapper smoke test should stop after 4 frames.');
assert(isfinite(result.metrics.finalPositionError), 'Follow wrapper final error must be finite.');
assert(isfile(result.paths.logPath), 'Missing follow wrapper log.');

disp('REAL_CAMERA_FOLLOW_CONTINUOUS_SMOKE_TEST_PASS');
end
