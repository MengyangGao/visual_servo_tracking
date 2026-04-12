function [uv, depth, pc] = project_points(worldPoints, T_wc, camera)
%SIMULATION_PROJECT_POINTS Project 3D world points into the image plane.
%
%   worldPoints is Nx3, T_wc maps camera coordinates into world coordinates.

if isempty(worldPoints)
    uv = zeros(0, 2);
    depth = zeros(0, 1);
    pc = zeros(0, 3);
    return;
end

worldPoints = double(worldPoints);
if size(worldPoints, 2) ~= 3
    error('project_points:BadPoints', 'worldPoints must be Nx3.');
end

T_cw = inv(T_wc);
pts = [worldPoints, ones(size(worldPoints, 1), 1)]';
pcHom = T_cw * pts;
pc = pcHom(1:3, :)';

depth = pc(:, 3);
uv = nan(size(worldPoints, 1), 2);
valid = depth > eps;
if any(valid)
    x = pc(valid, 1) ./ depth(valid);
    y = pc(valid, 2) ./ depth(valid);
    uv(valid, 1) = camera.fx * x + camera.cx;
    uv(valid, 2) = camera.fy * y + camera.cy;
end
end
