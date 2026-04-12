function result = run_real_camera_follow_continuous(varargin)
%RUN_REAL_CAMERA_FOLLOW_CONTINUOUS Legacy compatibility wrapper for follow mode.
%
% Prefer RUN_REAL_CAMERA_FOLLOW with RunMode="manual" or RunMode="auto".

if ~local_has_name(varargin, "RunMode")
    varargin = [varargin, {'RunMode', 'manual'}];
end
result = run_real_camera_follow(varargin{:});
end

function tf = local_has_name(args, name)
tf = false;
for i = 1:2:numel(args)
    if strcmpi(string(args{i}), string(name))
        tf = true;
        return;
    end
end
end
