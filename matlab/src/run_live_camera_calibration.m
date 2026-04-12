function results = run_live_camera_calibration(varargin)
%RUN_LIVE_CAMERA_CALIBRATION Capture a printed ChArUco board with webcam.

rootDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(rootDir, 'src')));

cfg = config(varargin{:});
cfg.realCamera.preCaptureDelaySec = max(cfg.realCamera.preCaptureDelaySec, 5);
results = live_camera_calibration(cfg);
end
