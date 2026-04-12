function detection = detect_target(img, colorName)
%SIMULATION_DETECT_TARGET Detect a colored target in a synthetic RGB image.

colorName = lower(string(colorName));
rgb = double(img);
switch colorName
    case "red"
        mask = rgb(:, :, 1) > 165 & rgb(:, :, 2) < 120 & rgb(:, :, 3) < 120;
    case "green"
        mask = rgb(:, :, 2) > 145 & rgb(:, :, 1) < 125 & rgb(:, :, 3) < 125;
    case "blue"
        mask = rgb(:, :, 3) > 150 & rgb(:, :, 1) < 140 & rgb(:, :, 2) < 140;
    case "yellow"
        mask = rgb(:, :, 1) > 170 & rgb(:, :, 2) > 140 & rgb(:, :, 3) < 120;
    otherwise
        error('detect_target:UnknownColor', 'Unsupported target color "%s".', colorName);
end

mask = imclose(mask, strel('disk', 2));
mask = bwareaopen(mask, 8);
stats = regionprops(mask, 'Area', 'Centroid', 'BoundingBox');

detection = struct();
detection.success = ~isempty(stats);
detection.mask = mask;

if detection.success
    [~, idx] = max([stats.Area]);
    detection.centroid = stats(idx).Centroid;
    detection.bbox = stats(idx).BoundingBox;
    detection.area = stats(idx).Area;
else
    detection.centroid = [NaN NaN];
    detection.bbox = [];
    detection.area = 0;
end
end
