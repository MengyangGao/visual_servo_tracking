function worldPoint = backproject_to_plane(pixel, T_wc, camera, planeZ)
%SIMULATION_BACKPROJECT_TO_PLANE Intersect a camera ray with a horizontal plane.

pixel = pixel(:);
if numel(pixel) ~= 2
    error('backproject_to_plane:BadPixel', 'pixel must be a 2-vector.');
end

rayCam = [(pixel(1) - camera.cx) / camera.fx;
          (pixel(2) - camera.cy) / camera.fy;
          1];

R_wc = T_wc(1:3, 1:3);
origin = T_wc(1:3, 4);
rayWorld = R_wc * rayCam;

if abs(rayWorld(3)) < 1e-9
    error('backproject_to_plane:ParallelRay', 'Camera ray is nearly parallel to the plane.');
end

scale = (planeZ - origin(3)) / rayWorld(3);
worldPoint = origin + scale * rayWorld;
worldPoint = worldPoint(:).';
end
