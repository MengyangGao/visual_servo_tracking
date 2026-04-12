function session = session_controls(fig, runMode, titleText)
%SESSION_CONTROLS Create auto/manual run controls for a figure.

if nargin < 2 || isempty(runMode)
    runMode = "auto";
end
if nargin < 3 || isempty(titleText)
    titleText = "Session";
end

runMode = normalize_run_mode(runMode);
session = struct();
session.fig = fig;
session.titleText = char(titleText);
session.mode = char(runMode);
session.manual = strcmp(runMode, "manual");
session.startKey = "SessionRunRequested";
session.stopKey = "SessionStopRequested";
session.statusText = [];
session.startButton = [];
session.stopButton = [];
session.waitForStart = @() local_wait_for_start(fig, session.startKey, session.stopKey);
session.shouldStop = @() local_should_stop(fig, session.stopKey);
session.requestStop = @() local_request_stop(fig, session.startKey, session.stopKey, session.statusText, session.startButton, session.stopButton, false);
session.cleanup = @() local_cleanup(fig, session.stopKey, session.statusText, session.startButton, session.stopButton);

if ~session.manual
    setappdata(fig, session.startKey, true);
    setappdata(fig, session.stopKey, false);
    return;
end

setappdata(fig, session.startKey, false);
setappdata(fig, session.stopKey, false);

statusText = uicontrol(fig, 'Style', 'text', ...
    'Units', 'normalized', ...
    'Position', [0.02 0.01 0.60 0.05], ...
    'String', sprintf('%s ready - press Start to run', session.titleText), ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', get(fig, 'Color'));
startButton = uicontrol(fig, 'Style', 'pushbutton', ...
    'Units', 'normalized', ...
    'Position', [0.65 0.01 0.14 0.055], ...
    'String', 'Start', ...
    'FontWeight', 'bold');
stopButton = uicontrol(fig, 'Style', 'pushbutton', ...
    'Units', 'normalized', ...
    'Position', [0.81 0.01 0.14 0.055], ...
    'String', 'Stop', ...
    'FontWeight', 'bold', ...
    'Enable', 'off');

set(startButton, 'Callback', @(src, evt) local_request_start(fig, session.startKey, session.stopKey, statusText, startButton, stopButton, session.titleText));
set(stopButton, 'Callback', @(src, evt) local_request_stop(fig, session.startKey, session.stopKey, statusText, startButton, stopButton, true));
set(fig, 'CloseRequestFcn', @(src, evt) local_close_request(fig, session.startKey, session.stopKey, statusText, startButton, stopButton, session.titleText));

session.statusText = statusText;
session.startButton = startButton;
session.stopButton = stopButton;
end

function tf = local_should_stop(fig, stopKey)
tf = ~isgraphics(fig) || local_get_flag(fig, stopKey);
end

function local_wait_for_start(fig, startKey, stopKey)
while isgraphics(fig) && ~local_get_flag(fig, startKey) && ~local_get_flag(fig, stopKey)
    drawnow;
    pause(0.05);
end
end

function local_request_start(fig, startKey, stopKey, statusText, startButton, stopButton, titleText)
if ~isgraphics(fig)
    return;
end
setappdata(fig, startKey, true);
setappdata(fig, stopKey, false);
if isgraphics(statusText)
    statusText.String = sprintf('%s running - press Stop to end', titleText);
end
if isgraphics(startButton)
    startButton.Enable = 'off';
end
if isgraphics(stopButton)
    stopButton.Enable = 'on';
end
drawnow;
end

function local_request_stop(fig, startKey, stopKey, statusText, startButton, stopButton, keepFigure)
if isgraphics(fig)
    setappdata(fig, stopKey, true);
    setappdata(fig, startKey, false);
end
if isgraphics(statusText)
    statusText.String = 'Stopping...';
end
if isgraphics(startButton)
    startButton.Enable = 'off';
end
if isgraphics(stopButton)
    stopButton.Enable = 'off';
end
drawnow;
if nargin >= 7 && keepFigure
    return;
end
if isgraphics(fig)
    try
        delete(fig);
    catch
    end
end
end

function local_close_request(fig, startKey, stopKey, statusText, startButton, stopButton, titleText)
if isgraphics(statusText)
    statusText.String = sprintf('Stopping %s...', titleText);
end
local_request_stop(fig, startKey, stopKey, statusText, startButton, stopButton, false);
end

function local_cleanup(fig, stopKey, statusText, startButton, stopButton)
if isgraphics(fig)
    setappdata(fig, stopKey, true);
end
if isgraphics(statusText)
    statusText.String = 'Finished';
end
if isgraphics(startButton)
    startButton.Enable = 'off';
end
if isgraphics(stopButton)
    stopButton.Enable = 'off';
end
end

function tf = local_get_flag(fig, key)
tf = false;
if isgraphics(fig) && isappdata(fig, key)
    tf = logical(getappdata(fig, key));
end
end
