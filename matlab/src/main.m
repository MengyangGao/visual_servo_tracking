function results = main(cfg)
%SIMULATION_MAIN Execute the Simulation tasks sequentially.

if nargin < 1 || isempty(cfg)
    cfg = config();
end

ensure_dir(cfg.resultsDir);
ensure_dir(cfg.paths.calibration);
ensure_dir(cfg.paths.fixedTracking);
ensure_dir(cfg.paths.eyeTracking);
ensure_dir(cfg.paths.ibvs);

logPath = fullfile(cfg.resultsDir, 'run_log.txt');
if isfile(logPath)
    delete(logPath);
end
diary(logPath);
logCleanup = onCleanup(@() diary('off')); %#ok<NASGU>

rng(cfg.randomSeed, 'twister');

fprintf('[Simulation] Run started: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('[Simulation] Log file: %s\n', logPath);
fprintf('[Simulation] Running T1: virtual calibration...\n');
results = struct();
results.config = cfg;
results.t1 = t1_virtual_calibration(cfg);
local_log_t1(results.t1);

fprintf('[Simulation] Running T2: fixed-camera position tracking...\n');
results.t2_fixed = t2_position_tracking(cfg, 'fixed', 'auto');
local_log_t2('fixed-camera', results.t2_fixed);

fprintf('[Simulation] Running T2: eye-in-hand position tracking...\n');
results.t2_eye = t2_position_tracking(cfg, 'eye-in-hand', 'auto');
local_log_t2('eye-in-hand', results.t2_eye);

fprintf('[Simulation] Running T3: IBVS square tracking...\n');
results.t3 = t3_ibvs_square(cfg, 'auto');
local_log_t3(results.t3);

summaryPath = fullfile(cfg.resultsDir, 'summary.mat');
if isfield(cfg, 'saveSummary') && cfg.saveSummary
    save(summaryPath, 'results', 'cfg');
    results.summaryPath = summaryPath;
    fprintf('[Simulation] Finished. Summary saved to %s\n', summaryPath);
else
    results.summaryPath = '';
    fprintf('[Simulation] Finished. Summary output disabled.\n');
end

results.logPath = logPath;
fprintf('[Simulation] Log stored at %s\n', logPath);
end

function local_log_t1(t1)
fprintf('[Summary][T1] boardType=%s, views=%d, meanReprojectionError=%.4f px\n', ...
    t1.metrics.boardType, t1.metrics.numViewsUsed, t1.metrics.meanReprojectionError);
trueK = t1.metrics.trueIntrinsics;
estK = t1.metrics.estimatedIntrinsics;
fprintf('[Summary][T1] fx/fy true=(%.2f, %.2f), estimated=(%.2f, %.2f)\n', ...
    trueK(1,1), trueK(2,2), estK(1,1), estK(2,2));
fprintf('[Summary][T1] printableBoard=%s\n', t1.paths.printableBoardPath);
end

function local_log_t2(modeLabel, t2)
fprintf('[Summary][T2 %s] finalError=%.6f m, rmse=%.6f m, detectionStartStep=%d, motionStartStep=%d\n', ...
    modeLabel, t2.metrics.finalPositionError, t2.metrics.rmsePositionError, ...
    t2.metrics.detectionStartStep, t2.metrics.motionStartStep);
fprintf('[Summary][T2 %s] finalEE=(%.4f, %.4f, %.4f) m\n', ...
    modeLabel, t2.metrics.finalEndEffectorPosition(1), t2.metrics.finalEndEffectorPosition(2), t2.metrics.finalEndEffectorPosition(3));
fprintf('[Summary][T2 %s] trajectory=%s\n', modeLabel, t2.paths.figures.trajectoryPath);
end

function local_log_t3(t3)
fprintf('[Summary][T3] finalFeatureError=%.6f, rmse=%.6f, finalCamera=(%.4f, %.4f, %.4f) m\n', ...
    t3.metrics.finalFeatureError, t3.metrics.rmseFeatureError, ...
    t3.metrics.finalCameraPosition(1), t3.metrics.finalCameraPosition(2), t3.metrics.finalCameraPosition(3));
fprintf('[Summary][T3] desiredCamera=(%.4f, %.4f, %.4f) m\n', ...
    t3.metrics.desiredCameraPosition(1), t3.metrics.desiredCameraPosition(2), t3.metrics.desiredCameraPosition(3));
fprintf('[Summary][T3] errorFigure=%s\n', t3.paths.figures.errorPath);
end
