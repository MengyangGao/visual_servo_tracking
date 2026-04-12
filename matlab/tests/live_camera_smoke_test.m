function live_camera_smoke_test()
%LIVE_CAMERA_SMOKE_TEST Smoke test for the real-camera ChArUco workflow.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(rootDir, 'src')));

cfg = config('showFigures', false, 'saveFigures', false, 'saveVideos', false);
board = charuco_board_asset(cfg);

assert(isfile(board.paths.png), 'Printable ChArUco PNG was not generated.');
assert(isfile(board.paths.pdf), 'Printable ChArUco PDF was not generated.');

[pts, used] = detectCharucoBoardPoints( ...
    board.image, ...
    board.patternDims, ...
    board.markerFamily, ...
    board.checkerSize, ...
    board.markerSize, ...
    MinMarkerID=board.minMarkerID, ...
    OriginCheckerColor=board.originCheckerColor, ...
    RefineCorners=true);

assert(used, 'Printable ChArUco board could not be detected.');
assert(size(pts, 1) == size(board.worldPoints, 1), 'Detected ChArUco point count mismatch.');

available = string(webcamlist);
assert(~isempty(available), 'No webcam was detected.');
idx = find(strcmpi(available, "MacBook Pro Camera"), 1);
if isempty(idx)
    camName = available(1);
else
    camName = available(idx);
end
cam = webcam(camName);
cleanup = onCleanup(@() delete(cam)); %#ok<NASGU>
frame = snapshot(cam);

assert(ndims(frame) == 3 && size(frame, 3) == 3, 'Webcam snapshot should be RGB.');
assert(~isempty(frame), 'Webcam snapshot is empty.');

disp('LIVE_CAMERA_SMOKE_TEST_PASS');
end
