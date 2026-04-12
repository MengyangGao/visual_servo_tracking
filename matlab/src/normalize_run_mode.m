function runMode = normalize_run_mode(runMode)
%NORMALIZE_RUN_MODE Normalize legacy and public run-mode labels.

if nargin < 1 || isempty(runMode)
    runMode = "auto";
else
    runMode = lower(string(runMode));
end

switch runMode
    case {"auto", "automatic", "finite"}
        runMode = "auto";
    case {"manual", "continuous"}
        runMode = "manual";
    otherwise
        error('normalize_run_mode:UnknownRunMode', 'Unsupported run mode "%s".', runMode);
end
end
