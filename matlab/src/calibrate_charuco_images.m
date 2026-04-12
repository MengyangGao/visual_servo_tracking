function calib = calibrate_charuco_images(images, board, imageSize, requiredDetections)
%CALIBRATE_CHARUCO_IMAGES Detect ChArUco corners and estimate intrinsics.

if nargin < 4
    requiredDetections = 1;
end

if iscell(images)
    imageList = images(:);
else
    imageList = squeeze(num2cell(images, [1 2 3]));
end

detectedPoints = {};
usedFrames = [];
for i = 1:numel(imageList)
    [pts, used] = detectCharucoBoardPoints( ...
        imageList{i}, ...
        board.patternDims, ...
        board.markerFamily, ...
        board.checkerSize, ...
        board.markerSize, ...
        MinMarkerID=board.minMarkerID, ...
        OriginCheckerColor=board.originCheckerColor, ...
        RefineCorners=true);

    if used && ~isempty(pts) && all(isfinite(pts(:)))
        detectedPoints{end + 1} = pts; %#ok<AGROW>
        usedFrames(end + 1) = i; %#ok<AGROW>
    end
end

if numel(detectedPoints) < requiredDetections
    error('calibrate_charuco_images:TooFewFrames', ...
        'Only %d valid ChArUco frames were detected; required at least %d.', ...
        numel(detectedPoints), requiredDetections);
end

numPoints = size(detectedPoints{1}, 1);
for i = 2:numel(detectedPoints)
    if size(detectedPoints{i}, 1) ~= numPoints
        error('calibrate_charuco_images:PointCountMismatch', ...
            'Detected frame %d has %d points, expected %d.', i, size(detectedPoints{i}, 1), numPoints);
    end
end

imagePoints = nan(numPoints, 2, numel(detectedPoints));
for i = 1:numel(detectedPoints)
    imagePoints(:, :, i) = detectedPoints{i};
end

worldPoints2D = patternWorldPoints("charuco-board", board.patternDims, board.checkerSize);
worldPoints3D = [worldPoints2D, zeros(size(worldPoints2D, 1), 1)];
cameraParams = estimateCameraParameters(imagePoints, worldPoints2D, 'ImageSize', imageSize);

perViewError = zeros(numel(detectedPoints), 1);
for i = 1:numel(detectedPoints)
    uvHat = worldToImage(cameraParams, cameraParams.RotationMatrices(:, :, i), ...
        cameraParams.TranslationVectors(i, :), worldPoints3D);
    perViewError(i) = mean(vecnorm(uvHat - detectedPoints{i}, 2, 2));
end

[estimatedK, distortionCoeffs] = local_extract_intrinsics(cameraParams);

calib = struct();
calib.cameraParams = cameraParams;
calib.metrics = struct();
calib.metrics.imageCount = numel(detectedPoints);
calib.metrics.usedFrames = usedFrames;
calib.metrics.perViewError = perViewError;
calib.metrics.meanReprojectionError = mean(perViewError);
calib.metrics.estimatedIntrinsics = estimatedK;
calib.metrics.distortionCoefficients = distortionCoeffs;
calib.worldPoints2D = worldPoints2D;
calib.worldPoints3D = worldPoints3D;
calib.imagePoints = imagePoints;
calib.imageSize = imageSize;
end

function [K, distortionCoeffs] = local_extract_intrinsics(cameraParams)
if isprop(cameraParams, 'IntrinsicMatrix')
    intrinsicsMatrix = cameraParams.IntrinsicMatrix;
    K = intrinsicsMatrix.';
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
