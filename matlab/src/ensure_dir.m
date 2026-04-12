function ensure_dir(folder)
%SIMULATION_ENSURE_DIR Create a directory if it does not already exist.

if ~exist(folder, 'dir')
    mkdir(folder);
end
end
