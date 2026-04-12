function [writer, pathUsed] = open_video_writer(basePath, frameRate)
%SIMULATION_OPEN_VIDEO_WRITER Open a video writer with a resilient fallback.

folder = fileparts(basePath);
if ~exist(folder, 'dir')
    mkdir(folder);
end

pathUsed = basePath;
try
    writer = VideoWriter(basePath, 'MPEG-4');
    writer.FrameRate = frameRate;
    open(writer);
catch
    [~, name] = fileparts(basePath);
    pathUsed = fullfile(folder, [name, '.avi']);
    writer = VideoWriter(pathUsed, 'Motion JPEG AVI');
    writer.FrameRate = frameRate;
    open(writer);
end
end
