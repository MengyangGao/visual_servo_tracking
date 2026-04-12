function savedPath = save_figure(figHandle, savedPath)
%SIMULATION_SAVE_FIGURE Save a figure to disk using exportgraphics.

folder = fileparts(savedPath);
if ~exist(folder, 'dir')
    mkdir(folder);
end

drawnow;
try
    set(figHandle, 'Color', 'w', 'InvertHardcopy', 'off');
    exportgraphics(figHandle, savedPath, 'Resolution', 240);
catch
    saveas(figHandle, savedPath);
end
end
