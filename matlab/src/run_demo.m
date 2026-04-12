function results = run_demo(varargin)
%RUN_DEMO Entry point for the simulation workflow.
%
%   results = RUN_DEMO() runs the full simulation workflow.
%   results = RUN_DEMO(... name-value overrides ...) forwards overrides to
%   the configuration helper.

rootDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(rootDir, 'src')));

cfg = config(varargin{:});
results = main(cfg);
end
