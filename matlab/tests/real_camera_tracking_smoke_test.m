function real_camera_tracking_smoke_test()
%REAL_CAMERA_TRACKING_SMOKE_TEST Smoke test for the real-camera board demos.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(rootDir, 'src')));

calibrationDir = fullfile(rootDir, 'results', 't1_live_calibration', 'calibration');
cameraParamsPath = fullfile(calibrationDir, 'cameraParams.mat');
imagePath = fullfile(calibrationDir, 'Image1.png');
if ~isfile(cameraParamsPath) || ~isfile(imagePath)
    fprintf('REAL_CAMERA_TRACKING_SMOKE_TEST_SKIPPED\n');
    return;
end

cfg = config('showFigures', false, 'saveFigures', false, 'saveVideos', false);
cfg.realCamera.followCount = 6;
cfg.realCamera.ibvsCount = 6;
cfg.realCamera.frameSequenceDir = calibrationDir;
cfg.realCamera.cameraParamsCandidates = {cameraParamsPath};

followResult = real_camera_tracking(cfg, 'follow', 'files');
assert(isfield(followResult.metrics, 'finalPositionError'), 'Follow result missing position metrics.');
assert(isfinite(followResult.metrics.finalPositionError), 'Follow final error must be finite.');
assert(strcmp(followResult.source, 'files'), 'Follow smoke test should use file sequence.');

ibvsResult = real_camera_tracking(cfg, 'ibvs', 'files');
assert(isfield(ibvsResult.metrics, 'finalFeatureError'), 'IBVS result missing feature metrics.');
assert(isfinite(ibvsResult.metrics.finalFeatureError), 'IBVS final error must be finite.');
assert(strcmp(ibvsResult.source, 'files'), 'IBVS smoke test should use file sequence.');

disp('REAL_CAMERA_TRACKING_SMOKE_TEST_PASS');
end
