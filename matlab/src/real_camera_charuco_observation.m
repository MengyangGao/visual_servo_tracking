function obs = real_camera_charuco_observation(frame, cameraParams, board)
%REAL_CAMERA_CHARUCO_OBSERVATION Detect a ChArUco board in a real frame.

obs = struct();
obs.success = false;
obs.points = [];
obs.used = false;
obs.bbox = [];
obs.bboxCorners = [];
obs.centerPixel = [NaN NaN];
obs.centerCamera = [NaN NaN NaN];
obs.rotation = [];
obs.translation = [];
obs.reprojectionError = NaN;
obs.undistortedFrame = frame;
obs.poseCameraParams = cameraParams;

if isempty(frame)
    return;
end

if nargin < 2 || isempty(cameraParams)
    poseFrame = frame;
    poseCameraParams = [];
else
    try
        [poseFrame, poseCameraParams] = undistortImage(frame, cameraParams);
    catch
        poseFrame = frame;
        poseCameraParams = cameraParams;
    end
end

obs.undistortedFrame = poseFrame;
obs.poseCameraParams = poseCameraParams;

[pts, used] = detectCharucoBoardPoints( ...
    poseFrame, ...
    board.patternDims, ...
    board.markerFamily, ...
    board.checkerSize, ...
    board.markerSize, ...
    MinMarkerID=board.minMarkerID, ...
    OriginCheckerColor=board.originCheckerColor, ...
    RefineCorners=true);

obs.used = used;
if ~used || isempty(pts) || any(~isfinite(pts(:)))
    return;
end

obs.success = true;
obs.points = pts;
obs.centerPixel = mean(pts, 1);
obs.bbox = local_axis_aligned_bbox(pts);
obs.bboxCorners = local_oriented_bbox(pts);

if isempty(poseCameraParams)
    return;
end

worldPoints = board.worldPoints;
if size(worldPoints, 2) == 2
    worldPoints3D = [worldPoints, zeros(size(worldPoints, 1), 1)];
else
    worldPoints3D = worldPoints;
end
if isprop(poseCameraParams, 'Intrinsics')
    intr = poseCameraParams.Intrinsics;
else
    intr = poseCameraParams;
end

try
    camExtrinsics = estimateExtrinsics(pts, worldPoints, intr);
    R = camExtrinsics.R;
    t = camExtrinsics.Translation;
    obs.rotation = R;
    obs.translation = t;
    centerWorld = mean(worldPoints3D, 1);
    obs.centerCamera = (R * centerWorld(:) + t(:)).';
    [~, fx, fy, cx, cy] = extract_intrinsics(poseCameraParams);
    t = t(:);
    pc = (R * worldPoints3D.' + t).';
    uvHat = [fx * pc(:, 1) ./ pc(:, 3) + cx, fy * pc(:, 2) ./ pc(:, 3) + cy];
    obs.reprojectionError = mean(vecnorm(uvHat - pts, 2, 2));
catch
    obs.reprojectionError = NaN;
end
end

function bbox = local_axis_aligned_bbox(points)
xMin = min(points(:, 1));
yMin = min(points(:, 2));
xMax = max(points(:, 1));
yMax = max(points(:, 2));
bbox = [xMin, yMin, xMax - xMin, yMax - yMin];
end

function corners = local_oriented_bbox(points)
center = mean(points, 1);
demeaned = points - center;
covMat = cov(demeaned);
[eigVecs, eigVals] = eig(covMat); %#ok<ASGLU>
[~, order] = sort(diag(eigVals), 'descend');
axesMat = eigVecs(:, order);
proj = demeaned * axesMat;
min1 = min(proj(:, 1));
max1 = max(proj(:, 1));
min2 = min(proj(:, 2));
max2 = max(proj(:, 2));

raw = [ ...
    min1, min2;
    max1, min2;
    max1, max2;
    min1, max2];
corners = raw * axesMat.' + center;
corners = local_order_image_order(corners);
end

function corners = local_order_image_order(corners)
[~, order] = sortrows(corners, [2 1]);
top = corners(order(1:2), :);
bottom = corners(order(3:4), :);

if top(1, 1) <= top(2, 1)
    tl = top(1, :);
    tr = top(2, :);
else
    tl = top(2, :);
    tr = top(1, :);
end

if bottom(1, 1) <= bottom(2, 1)
    bl = bottom(1, :);
    br = bottom(2, :);
else
    bl = bottom(2, :);
    br = bottom(1, :);
end

corners = [tl; tr; br; bl];
end
