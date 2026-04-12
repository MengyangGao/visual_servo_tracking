function board = run_charuco_board_asset(varargin)
%RUN_CHARUCO_BOARD_ASSET Generate the printable ChArUco board asset.

rootDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(rootDir, 'src')));

cfg = config(varargin{:});
board = charuco_board_asset(cfg);
end
