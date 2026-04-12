function draw_camera_frame(ax, T, scale, label, lineWidth)
%DRAW_CAMERA_FRAME Draw a small camera coordinate triad in 3D.

if nargin < 3 || isempty(scale)
    scale = 0.05;
end
if nargin < 4
    label = '';
end
if nargin < 5 || isempty(lineWidth)
    lineWidth = 1.8;
end

origin = T(1:3, 4).';
axesDirs = T(1:3, 1:3) * scale;
colors = [0.84 0.18 0.18; 0.18 0.65 0.28; 0.16 0.38 0.82];
for i = 1:3
    endPt = origin + axesDirs(:, i).';
    plot3(ax, [origin(1), endPt(1)], [origin(2), endPt(2)], [origin(3), endPt(3)], ...
        '-', 'Color', colors(i, :), 'LineWidth', lineWidth);
end
plot3(ax, origin(1), origin(2), origin(3), 'o', ...
    'MarkerSize', max(5, round(5 * lineWidth)), ...
    'MarkerFaceColor', [0.1 0.1 0.1], ...
    'MarkerEdgeColor', 'w');
if ~isempty(label)
    labelPos = origin + 1.08 * axesDirs(:, 1).' + 0.02 * scale * [0 0 1];
    text(ax, labelPos(1), labelPos(2), labelPos(3), [' ', label], ...
        'FontWeight', 'bold', ...
        'FontSize', 8, ...
        'Color', [0.12 0.12 0.12], ...
        'BackgroundColor', [1 1 1], ...
        'Margin', 1, ...
        'VerticalAlignment', 'middle', ...
        'HorizontalAlignment', 'left');
end
end
