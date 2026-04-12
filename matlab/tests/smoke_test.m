function smoke_test()
%SMOKE_TEST Batch smoke test for the simulation workflow.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(rootDir, 'src')));

cfg = config('fastMode', true, 'showFigures', false, 'saveFigures', true, 'saveVideos', true, 'saveSummary', false);
results = main(cfg);

assert(isempty(results.summaryPath), 'Summary MAT file should be disabled in the public workflow.');
assert(isfile(results.logPath), 'Missing run log.');
assert(isfile(results.t1.paths.printableBoardPath), 'Missing printable ChArUco board.');
assert(isfile(results.t1.paths.figures.boardPath), 'Missing T1 board figure.');
assert(isfile(results.t1.paths.figures.imagePath), 'Missing T1 image figure.');
assert(isfile(results.t1.paths.figures.intrinsicsPath), 'Missing T1 intrinsics figure.');
assert(isfile(results.t1.paths.video), 'Missing T1 video.');
assert(isfile(results.t2_fixed.paths.figures.trajectoryPath), 'Missing T2 fixed trajectory figure.');
assert(isfile(results.t2_fixed.paths.figures.errorPath), 'Missing T2 fixed error figure.');
assert(isfile(results.t2_fixed.paths.figures.jointPath), 'Missing T2 fixed joint figure.');
assert(isfile(results.t2_fixed.paths.figures.supportPath), 'Missing T2 fixed support figure.');
assert(isfile(results.t2_fixed.paths.video), 'Missing T2 fixed video.');
assert(isfile(results.t2_eye.paths.figures.trajectoryPath), 'Missing T2 eye-in-hand trajectory figure.');
assert(isfile(results.t2_eye.paths.figures.errorPath), 'Missing T2 eye-in-hand error figure.');
assert(isfile(results.t2_eye.paths.figures.jointPath), 'Missing T2 eye-in-hand joint figure.');
assert(isfile(results.t2_eye.paths.figures.supportPath), 'Missing T2 eye-in-hand support figure.');
assert(isfile(results.t2_eye.paths.video), 'Missing T2 eye-in-hand video.');
assert(isfile(results.t3.paths.figures.errorPath), 'Missing T3 error figure.');
assert(isfile(results.t3.paths.figures.featurePath), 'Missing T3 feature figure.');
assert(isfile(results.t3.paths.figures.jointPath), 'Missing T3 joint figure.');
assert(isfile(results.t3.paths.figures.depthPath), 'Missing T3 depth figure.');
assert(isfile(results.t3.paths.video), 'Missing T3 video.');

assert(results.t1.metrics.meanReprojectionError < 2.0, 'T1 reprojection error too large.');
assert(results.t2_fixed.metrics.rmsePositionError < 0.08, 'T2 fixed-camera tracking is too inaccurate.');
assert(results.t2_eye.metrics.rmsePositionError < 0.08, 'T2 eye-in-hand tracking is too inaccurate.');
assert(results.t3.metrics.finalFeatureError < 0.05, 'T3 feature error did not converge sufficiently.');

disp('SMOKE_TEST_PASS');
end
