function corners = square_corners(center, sideLength)
%SIMULATION_SQUARE_CORNERS Return the 4 corners of a square on the z-plane.

center = center(:).';
half = sideLength / 2;
corners = [center(1) - half, center(2) - half, center(3);
           center(1) + half, center(2) - half, center(3);
           center(1) + half, center(2) + half, center(3);
           center(1) - half, center(2) + half, center(3)];
end
