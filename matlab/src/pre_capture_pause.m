function pre_capture_pause(seconds, label)
%PRE_CAPTURE_PAUSE Optional countdown before camera capture starts.

if nargin < 1 || isempty(seconds) || seconds <= 0
    return;
end
if nargin < 2 || isempty(label)
    label = 'Capture';
end

seconds = max(0, ceil(seconds));
fprintf('[%s] Starting capture in %d seconds...\n', label, seconds);
for k = seconds:-1:1
    fprintf('[%s] %d...\n', label, k);
    pause(1);
end
fprintf('[%s] Capture starts now.\n', label);
end
