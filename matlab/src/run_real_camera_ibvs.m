function result = run_real_camera_ibvs(varargin)
%RUN_REAL_CAMERA_IBVS Run the real-camera board-based IBVS demo.

rootDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(rootDir, 'src')));

[configArgs, taskOpts] = local_split_task_options(varargin{:});
cfg = config(configArgs{:});
if isempty(taskOpts.preCaptureDelaySec)
    cfg.realCamera.preCaptureDelaySec = max(cfg.realCamera.preCaptureDelaySec, 5);
else
    cfg.realCamera.preCaptureDelaySec = taskOpts.preCaptureDelaySec;
end
if ~isempty(taskOpts.maxFrames)
    cfg.realCamera.ibvsCount = taskOpts.maxFrames;
end
result = real_camera_tracking(cfg, 'ibvs', taskOpts.sourceMode, taskOpts.runMode);
end

function [configArgs, taskOpts] = local_split_task_options(varargin)
taskOpts = struct();
taskOpts.sourceMode = 'auto';
taskOpts.runMode = 'auto';
taskOpts.preCaptureDelaySec = [];
taskOpts.maxFrames = [];
configArgs = {};

if isempty(varargin)
    return;
end
if mod(numel(varargin), 2) ~= 0
    error('run_real_camera_ibvs:BadArguments', 'Arguments must be provided as name-value pairs.');
end

for i = 1:2:numel(varargin)
    key = string(varargin{i});
    value = varargin{i + 1};
    switch lower(key)
        case "sourcemode"
            taskOpts.sourceMode = value;
        case "runmode"
            taskOpts.runMode = value;
        case "precapturedelaysec"
            taskOpts.preCaptureDelaySec = value;
        case "maxframes"
            taskOpts.maxFrames = value;
        otherwise
            configArgs{end + 1} = varargin{i}; %#ok<AGROW>
            configArgs{end + 1} = value; %#ok<AGROW>
    end
end
end
