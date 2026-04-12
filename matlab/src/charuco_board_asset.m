function board = charuco_board_asset(cfg, outputDir, fileStem)
%CHARUCO_BOARD_ASSET Generate a printable ChArUco board asset in assets/.

if nargin < 2 || isempty(outputDir)
    if isfield(cfg, 'paths') && isfield(cfg.paths, 'assets') && ~isempty(cfg.paths.assets)
        outputDir = cfg.paths.assets;
    else
        outputDir = cfg.paths.liveCalibration;
    end
end
if nargin < 3 || isempty(fileStem)
    fileStem = 'charuco_board_printable';
end

ensure_dir(outputDir);

spec = cfg.realCamera.charuco;

board = struct();
board.patternDims = spec.patternDims;
board.markerFamily = spec.markerFamily;
board.checkerSize = spec.checkerSize;
board.markerSize = spec.markerSize;
board.imageSize = spec.imageSize;
board.marginPx = spec.marginPx;
board.minMarkerID = spec.minMarkerID;
board.originCheckerColor = spec.originCheckerColor;
board.boardPhysicalSizeMm = [spec.patternDims(2) * spec.checkerSize, spec.patternDims(1) * spec.checkerSize] * 1000;
board.boardPhysicalSizeM = board.boardPhysicalSizeMm / 1000;
board.worldPoints = patternWorldPoints("charuco-board", spec.patternDims, spec.checkerSize);
board.numPoints = size(board.worldPoints, 1);
board.boardCornersWorld = [ ...
    0, 0, 0;
    board.boardPhysicalSizeM(1), 0, 0;
    board.boardPhysicalSizeM(1), board.boardPhysicalSizeM(2), 0;
    0, board.boardPhysicalSizeM(2), 0];
board.boardCenterWorld = mean(board.boardCornersWorld, 1);
board.paths = struct();
board.paths.png = fullfile(outputDir, sprintf('%s.png', fileStem));
board.paths.pdf = fullfile(outputDir, sprintf('%s.pdf', fileStem));

board.image = generateCharucoBoard( ...
    spec.imageSize, ...
    spec.patternDims, ...
    spec.markerFamily, ...
    spec.checkerSizePx, ...
    spec.markerSizePx, ...
    MarginSize=1, ...
    MinMarkerID=spec.minMarkerID, ...
    OriginCheckerColor=spec.originCheckerColor);

imwrite(board.image, board.paths.png);
local_write_pdf(board.image, board.boardPhysicalSizeMm, board.paths.pdf, spec.marginPx);
board.textureRgb = repmat(board.image, 1, 1, 3);
end

function local_write_pdf(boardImage, boardPhysicalSizeMm, pdfPath, marginPx)
boardWidthMm = boardPhysicalSizeMm(1);
boardHeightMm = boardPhysicalSizeMm(2);
pageMarginMm = max(8, round(marginPx / 12));
pageWidthMm = boardWidthMm + 2 * pageMarginMm;
pageHeightMm = boardHeightMm + 2 * pageMarginMm;

pageWidthCm = pageWidthMm / 10;
pageHeightCm = pageHeightMm / 10;
boardWidthCm = boardWidthMm / 10;
boardHeightCm = boardHeightMm / 10;
pageMarginCm = pageMarginMm / 10;

fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'centimeters', ...
    'Position', [2 2 pageWidthCm pageHeightCm]);
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [pageWidthCm pageHeightCm], ...
    'PaperPosition', [0 0 pageWidthCm pageHeightCm]);

ax = axes('Parent', fig, 'Units', 'centimeters', ...
    'Position', [pageMarginCm, pageMarginCm, boardWidthCm, boardHeightCm]);
imshow(boardImage, 'Parent', ax, 'InitialMagnification', 'fit');
axis(ax, 'off');

drawnow;
print(fig, pdfPath, '-dpdf', '-painters');
close(fig);
end
