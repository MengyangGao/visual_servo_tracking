function T_wc = lookat_tform(cameraPosition, targetPoint, upVector)
%SIMULATION_LOOKAT_TFORM Build a camera pose that looks from cameraPosition to targetPoint.
%
%   The returned transform maps camera coordinates into world coordinates.

cameraPosition = cameraPosition(:);
targetPoint = targetPoint(:);
upVector = upVector(:);

forward = targetPoint - cameraPosition;
forwardNorm = norm(forward);
if forwardNorm < eps
    error('lookat_tform:DegeneratePose', 'Camera position and target must differ.');
end
forward = forward / forwardNorm;

up = upVector / norm(upVector);
if abs(dot(forward, up)) > 0.95
    if abs(forward(3)) < 0.95
        up = [0; 0; 1];
    else
        up = [0; 1; 0];
    end
end

right = cross(up, forward);
right = right / norm(right);
trueUp = cross(forward, right);

R_wc = [right, trueUp, forward];
T_wc = eye(4);
T_wc(1:3, 1:3) = R_wc;
T_wc(1:3, 4) = cameraPosition;
end
