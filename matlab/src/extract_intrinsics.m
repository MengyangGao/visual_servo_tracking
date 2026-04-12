function [K, fx, fy, cx, cy] = extract_intrinsics(cameraParams)
%SIMULATION_EXTRACT_INTRINSICS Convert camera parameters to a 3x3 intrinsic matrix.

if isprop(cameraParams, 'Intrinsics') && ~isempty(cameraParams.Intrinsics)
    intr = cameraParams.Intrinsics;
    fx = intr.FocalLength(1);
    fy = intr.FocalLength(2);
    cx = intr.PrincipalPoint(1);
    cy = intr.PrincipalPoint(2);
    K = [fx, 0, cx;
         0, fy, cy;
         0, 0, 1];
elseif isprop(cameraParams, 'IntrinsicMatrix')
    K = cameraParams.IntrinsicMatrix';
    fx = K(1, 1);
    fy = K(2, 2);
    cx = K(1, 3);
    cy = K(2, 3);
elseif isprop(cameraParams, 'FocalLength') && isprop(cameraParams, 'PrincipalPoint')
    fx = cameraParams.FocalLength(1);
    fy = cameraParams.FocalLength(2);
    cx = cameraParams.PrincipalPoint(1);
    cy = cameraParams.PrincipalPoint(2);
    K = [fx, 0, cx;
         0, fy, cy;
         0, 0, 1];
else
    error('extract_intrinsics:UnsupportedType', 'Unsupported camera parameters type.');
end
end
