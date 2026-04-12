function [img, details] = render_camera_view(T_wc, camera, scene)
%SIMULATION_RENDER_CAMERA_VIEW Render a synthetic RGB camera view from the scene.

imageSize = camera.imageSize;
h = imageSize(1);
w = imageSize(2);
img = uint8(zeros(h, w, 3));
for c = 1:3
    img(:, :, c) = scene.backgroundColor(c);
end

details = struct();
details.objects = struct([]);

if ~isfield(scene, 'objects')
    return;
end

for i = 1:numel(scene.objects)
    obj = scene.objects(i);
    switch lower(obj.kind)
        case 'sphere'
            [uv, depth] = project_points(obj.position, T_wc, camera);
            if ~isfinite(depth) || depth <= 0 || any(~isfinite(uv))
                continue;
            end
            radiusPx = max(2, round(camera.fx * obj.radius / depth));
            circle = [uv(1), uv(2), 2 * radiusPx];
            img = insertShape(img, 'FilledCircle', circle, 'Color', double(obj.color), 'Opacity', 1);
            img = insertShape(img, 'Circle', circle, 'Color', [25 25 25], 'LineWidth', 1);
            details.objects(i).name = obj.name;
            details.objects(i).center = uv;
            details.objects(i).depth = depth;
        case 'square'
            [uv, depth] = project_points(obj.corners, T_wc, camera);
            if any(~isfinite(uv(:))) || any(depth <= 0)
                continue;
            end
            poly = reshape(uv.', 1, []);
            img = insertShape(img, 'FilledPolygon', poly, 'Color', double(obj.fillColor), 'Opacity', 0.85);
            img = insertShape(img, 'Polygon', poly, 'Color', double(obj.edgeColor), 'LineWidth', 2);
            if isfield(obj, 'markCorners') && obj.markCorners
                for k = 1:size(uv, 1)
                    img = insertShape(img, 'FilledCircle', [uv(k, 1), uv(k, 2), 8], ...
                        'Color', double(obj.cornerColor), 'Opacity', 1);
                end
            end
            details.objects(i).name = obj.name;
            details.objects(i).corners = uv;
            details.objects(i).depth = depth;
        otherwise
            error('render_camera_view:UnknownObject', 'Unknown scene object kind "%s".', obj.kind);
    end
end

img = uint8(min(max(img, 0), 255));
end
