function [worldPoints2D, worldPoints3D] = checkerboard_points(rows, cols, squareSize, center)
%SIMULATION_CHECKERBOARD_POINTS Generate planar calibration points.

if nargin < 4 || isempty(center)
    center = [0 0 0];
end

[xGrid, yGrid] = meshgrid(0:cols - 1, 0:rows - 1);
xGrid = (xGrid - mean(xGrid(:))) * squareSize + center(1);
yGrid = (yGrid - mean(yGrid(:))) * squareSize + center(2);

worldPoints2D = [xGrid(:), yGrid(:)];
worldPoints3D = [worldPoints2D, center(3) * ones(size(worldPoints2D, 1), 1)];
end
