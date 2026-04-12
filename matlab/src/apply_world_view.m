function apply_world_view(ax, cfg, scene)
%APPLY_WORLD_VIEW Set a consistent front-facing 3D visualization view.

mode = "front";
if nargin >= 2 && isstruct(cfg) && isfield(cfg, 'visualization') && isfield(cfg.visualization, 'worldViewMode')
    mode = lower(string(cfg.visualization.worldViewMode));
end

switch mode
    case {"front", "macbook-front", "front-facing", "frontview"}
        if nargin < 3 || isempty(scene) || ~isfield(scene, 'tableX') || ~isfield(scene, 'tableY') || ~isfield(scene, 'tableZ')
            view(ax, 3);
            return;
        end

        xCenter = mean(scene.tableX);
        yCenter = mean(scene.tableY);
        zBase = scene.tableZ;
        spanX = diff(scene.tableX);
        spanY = diff(scene.tableY);
        span = max(spanX, spanY);
        camTarget = [xCenter, yCenter, zBase + 0.25];
        camPos = [xCenter, scene.tableY(1) - 1.6 * span, zBase + 0.60];

        camtarget(ax, camTarget);
        campos(ax, camPos);
        camup(ax, [0 0 1]);
        camproj(ax, 'orthographic');
        axis(ax, 'vis3d');
    otherwise
        view(ax, 3);
end
end
