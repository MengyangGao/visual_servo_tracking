function cfg = config(varargin)
%SIMULATION_CONFIG Build default configuration for the simulation workflow.

rootDir = fileparts(fileparts(mfilename('fullpath')));

cfg = struct();
cfg.rootDir = rootDir;
cfg.resultsDir = fullfile(rootDir, 'results');
cfg.paths = struct();
cfg.paths.assets = fullfile(rootDir, 'assets');
cfg.showFigures = usejava('desktop');
cfg.saveFigures = true;
cfg.saveVideos = true;
cfg.saveSummary = false;
cfg.fastMode = false;
cfg.randomSeed = 42;
cfg.videoFrameRate = 10;

cfg.camera = struct();
cfg.camera.imageSize = [480 640];
cfg.camera.fx = 900;
cfg.camera.fy = 900;
cfg.camera.cx = cfg.camera.imageSize(2) / 2;
cfg.camera.cy = cfg.camera.imageSize(1) / 2;
cfg.camera.distortion = zeros(1, 5);

cfg.robot = local_make_robot();
cfg.eeName = cfg.robot.BodyNames{end};

cfg.scene = struct();
cfg.scene.tableZ = 0;
cfg.scene.tableX = [0.05 0.65];
cfg.scene.tableY = [-0.35 0.35];
cfg.scene.tableColor = uint8([235 225 210]);
cfg.scene.backgroundColor = uint8([248 248 248]);

cfg.visualization = struct();
cfg.visualization.worldViewMode = "front";
cfg.visualization.worldBadgePosition = [0.66 0.90];
cfg.visualization.imageBadgePosition = [0.02 0.90];

cfg.calibration = struct();
cfg.calibration.boardRows = 6;
cfg.calibration.boardCols = 8;
cfg.calibration.squareSize = 0.03;
cfg.calibration.numViews = 24;
cfg.calibration.noiseStdPx = 0.35;
cfg.calibration.boardCenter = [0.33 0.00 0.00];
cfg.calibration.boardNormal = [0 0 1];
cfg.calibration.cameraPosition = [0.38 -0.05 0.72];
cfg.calibration.lookAtTarget = [0.33 0.00 0.00];
cfg.calibration.upVector = [0 1 0];
cfg.calibration.charuco = struct();
cfg.calibration.charuco.patternDims = [7 5];
cfg.calibration.charuco.markerFamily = "DICT_4X4_1000";
cfg.calibration.charuco.checkerSize = 0.03;
cfg.calibration.charuco.markerSize = 0.0225;
cfg.calibration.charuco.textureImageSize = [2942 2102];
cfg.calibration.charuco.textureCheckerSizePx = 420;
cfg.calibration.charuco.textureMarkerSizePx = 315;
cfg.calibration.charuco.originCheckerColor = "black";
cfg.calibration.charuco.minMarkerID = 0;

cfg.fixedCamera = struct();
cfg.fixedCamera.position = [0.34 0.00 0.90];
cfg.fixedCamera.target = [0.34 0.00 0.00];
cfg.fixedCamera.upVector = [0 1 0];

cfg.t2 = struct();
cfg.t2.numSteps = 60;
cfg.t2.dt = 0.08;
cfg.t2.commandAlpha = 0.22;
cfg.t2.hoverHeight = 0.45;
cfg.t2.targetRadius = 0.03;
cfg.t2.distractorRadius = 0.022;
cfg.t2.targetColorName = 'red';
cfg.t2.targetPathCenter = [0.34 0.00 0.00];
cfg.t2.targetPathRadius = [0.08 0.05];
cfg.t2.targetPathWiggle = [0.015 0.01];
cfg.t2.targetPathPhase = pi / 6;
cfg.t2.staticLeadInSteps = 24;
cfg.t2.detectionHoldSteps = 3;
cfg.t2.motionStartTolerance = 0.022;
cfg.t2.motionStartMaxSteps = 24;
cfg.t2.motionDurationSec = cfg.t2.numSteps * cfg.t2.dt;
cfg.t2.ikWeights = [0.25 0.25 0.25 1 1 1];
cfg.t2.limitsPadding = 0.03;
cfg.t2.initialHover = [0.26 -0.14 cfg.t2.hoverHeight];
cfg.t2.eyeInHandCamera = local_scaled_camera(cfg.camera, 0.62, [540 720]);
cfg.t2.distractors = [ ...
    struct('name', 'blue_1', 'position', [0.22 0.18 0], 'radius', cfg.t2.distractorRadius, 'color', uint8([60 100 220])), ...
    struct('name', 'green_1', 'position', [0.48 -0.14 0], 'radius', cfg.t2.distractorRadius, 'color', uint8([40 170 90])), ...
    struct('name', 'yellow_1', 'position', [0.52 0.16 0], 'radius', cfg.t2.distractorRadius, 'color', uint8([230 180 40])) ...
];

cfg.t3 = struct();
cfg.t3.numSteps = 80;
cfg.t3.dt = 0.05;
cfg.t3.squareCenter = [0.34 0.02 0.00];
cfg.t3.squareSide = 0.12;
cfg.t3.initialCameraPosition = [0.49 -0.16 0.58];
cfg.t3.desiredCameraPosition = [0.29 0.02 0.655];
cfg.t3.upVector = [0 1 0];
cfg.t3.ibvsGain = 0.85;
cfg.t3.interactionDamping = 1e-4;
cfg.t3.jacobianDamping = 1e-4;
cfg.t3.featureTolerance = 0.012;

cfg.paths.calibration = fullfile(cfg.resultsDir, 't1_calibration');
cfg.paths.fixedTracking = fullfile(cfg.resultsDir, 't2_fixed_camera_tracking');
cfg.paths.eyeTracking = fullfile(cfg.resultsDir, 't2_eye_in_hand_tracking');
cfg.paths.ibvs = fullfile(cfg.resultsDir, 't3_ibvs_square_tracking');
cfg.paths.liveCalibration = fullfile(cfg.resultsDir, 't1_live_calibration');
cfg.paths.liveCalibrationInputs = fullfile(cfg.paths.liveCalibration, 'calibration');
cfg.paths.realFixedTracking = fullfile(cfg.resultsDir, 't2_real_camera_follow');
cfg.paths.realIbvs = fullfile(cfg.resultsDir, 't3_real_camera_ibvs');

cfg.realCamera = struct();
cfg.realCamera.cameraNamePreference = ["MacBook Pro Camera", "Air Camera"];
cfg.realCamera.captureCount = 24;
cfg.realCamera.captureIntervalSec = 0.8;
cfg.realCamera.requiredDetections = 12;
cfg.realCamera.preview = true;
cfg.realCamera.saveRawFrames = true;
cfg.realCamera.preCaptureDelaySec = 0;
cfg.realCamera.followCount = 24;
cfg.realCamera.followIntervalSec = 0.15;
cfg.realCamera.followProxyScale = 0.85;
cfg.realCamera.ibvsCount = 24;
cfg.realCamera.ibvsIntervalSec = 0.15;
cfg.realCamera.ibvsGain = 0.75;
cfg.realCamera.historyLimit = 600;
cfg.realCamera.cameraParamsCandidates = { ...
    fullfile(rootDir, 'assets', 'cameraParams.mat'), ...
    fullfile(cfg.paths.liveCalibrationInputs, 'cameraParams.mat'), ...
    fullfile(cfg.paths.liveCalibration, 'camera_params.mat'), ...
    fullfile(rootDir, 'cameraParams.mat') ...
};
cfg.realCamera.frameSequenceDir = cfg.paths.liveCalibrationInputs;
cfg.realCamera.frameSequencePattern = "Image*.png";
cfg.realCamera.charuco = struct();
cfg.realCamera.charuco.patternDims = [7 5];
cfg.realCamera.charuco.markerFamily = "DICT_4X4_1000";
cfg.realCamera.charuco.checkerSize = 0.03;
cfg.realCamera.charuco.markerSize = 0.0225;
cfg.realCamera.charuco.imageSize = [2942 2102];
cfg.realCamera.charuco.marginPx = 120;
cfg.realCamera.charuco.checkerSizePx = 420;
cfg.realCamera.charuco.markerSizePx = 315;
cfg.realCamera.charuco.originCheckerColor = "black";
cfg.realCamera.charuco.minMarkerID = 0;

cfg = apply_overrides(cfg, varargin{:});

if cfg.fastMode
    cfg.calibration.numViews = min(cfg.calibration.numViews, 6);
    cfg.t2.numSteps = min(cfg.t2.numSteps, 40);
    cfg.t3.numSteps = min(cfg.t3.numSteps, 55);
    cfg.calibration.noiseStdPx = min(cfg.calibration.noiseStdPx, 0.25);
end

function robot = local_make_robot()
robot = loadrobot('puma560');
robot.DataFormat = 'row';
robot.Gravity = [0 0 -9.81];
end

function camera = local_scaled_camera(baseCamera, focalScale, imageSize)
camera = baseCamera;
camera.imageSize = imageSize;
camera.fx = baseCamera.fx * focalScale;
camera.fy = baseCamera.fy * focalScale;
camera.cx = imageSize(2) / 2;
camera.cy = imageSize(1) / 2;
if isfield(camera, 'distortion')
    camera.distortion = baseCamera.distortion;
end
end
end

function cfg = apply_overrides(cfg, varargin)
if isempty(varargin)
    return;
end

if numel(varargin) == 1 && isstruct(varargin{1})
    overrides = varargin{1};
else
    if mod(numel(varargin), 2) ~= 0
        error('config:BadOverrides', 'Overrides must be name-value pairs.');
    end
    overrides = struct();
    for k = 1:2:numel(varargin)
        overrides.(varargin{k}) = varargin{k + 1};
    end
end

fields = fieldnames(overrides);
for i = 1:numel(fields)
    cfg.(fields{i}) = overrides.(fields{i});
end
end
