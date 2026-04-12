function [cameraParams, sourcePath] = load_camera_params(cfg)
%LOAD_CAMERA_PARAMS Load a saved MATLAB cameraParameters object.

sourcePath = '';
cameraParams = [];

if isfield(cfg, 'realCamera') && isfield(cfg.realCamera, 'cameraParamsCandidates')
    candidates = cfg.realCamera.cameraParamsCandidates;
else
    candidates = {};
end

for i = 1:numel(candidates)
    candidate = candidates{i};
    if isstring(candidate) || ischar(candidate)
        candidate = char(candidate);
    end
    if isfile(candidate)
        loaded = load(candidate);
        names = fieldnames(loaded);
        if isfield(loaded, 'cameraParams')
            cameraParams = loaded.cameraParams;
        elseif ~isempty(names)
            cameraParams = loaded.(names{1});
        else
            error('load_camera_params:EmptyFile', 'Camera parameter file "%s" is empty.', candidate);
        end
        sourcePath = candidate;
        return;
    end
end

error('load_camera_params:NotFound', 'No camera parameter file was found in the configured search paths.');
end
