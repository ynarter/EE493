% Parameters 
range_resolution = 0.5; % Range resolution (example value in meters)
angle_resolution = 0.1; % Angle resolution (example value in radians)
noise_min = 0.005; % Minimum noise level
noise_max = 0.5;  % Maximum noise level

% Define Gaussian scaling parameters
range_sigma = 2; % Spread of the Gaussian in range (meters)
angle_sigma = 5; % Spread of the Gaussian in angle (degrees)

% Generate range and angle grids
range_bins = 128; % Number of range bins
angle_bins = 128; % Number of angle bins
range = linspace(0, 25, range_bins); % Range values (positive distances only)
angles = linspace(-60, 60, angle_bins); % Angles in degrees

% Convert to meshgrid
[Range, Angle] = meshgrid(range, angles);

% Initialize parameters for dataset generation
num_maps = 1000; % Number of maps to generate
labels = zeros(1, num_maps); % Label array (0: noise-only, 1: target-present)
target_types = zeros(1, num_maps); % 0: Absent, 1: Point-like, 2: Extended
maps = zeros(angle_bins, range_bins, num_maps); % Store maps

% Random number generator seed for reproducibility
rng(42);

% Create a folder to save the maps and labels
output_folder = 'generated_maps';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for i = 1:num_maps
    % Randomly decide whether a target is present
    target_present = rand > 0.5; % 50% chance of target presence
    labels = target_present;

    % Initialize map with noise
    noise_level = noise_min + (noise_max - noise_min) * rand;
    map = abs(noise_level * randn(angle_bins, range_bins));

    if target_present
        % Randomly select a center for the target
        target_range = 5 + 20 * rand; % Between 5 and 25 meters
        target_angle = -45 + 90 * rand; % Between -45 and 45 degrees

        % Decide if the target is extended or point-like
        is_extended = rand > 0.5;

        if is_extended
            % Extended Gaussian target
            range_gaussian = exp(-((Range - target_range).^2) / (2 * range_sigma^2));
            angle_gaussian = exp(-((Angle - target_angle).^2) / (2 * angle_sigma^2));
            target = range_gaussian .* angle_gaussian; % 2D Gaussian
            target_types(i) = 2; % Extended target
            labels = 1;
        else
            % Point-like Gaussian target
            small_range_sigma = 0.5;
            small_angle_sigma = 1;
            range_gaussian = exp(-((Range - target_range).^2) / (2 * small_range_sigma^2));
            angle_gaussian = exp(-((Angle - target_angle).^2) / (2 * small_angle_sigma^2));
            target = range_gaussian .* angle_gaussian;
            target_types(i) = 1; % Point-like target
            labels = 1;
        end
        
        target = target / max(target(:));
        % Add the target to the map
        map = map + target;
    else
        % Noise-only target type
        target_types(i) = 0; % Absent target
        labels = 0;
    end

    % Normalize the map
    map = map / max(map(:));

    % Apply circular masking to simulate realistic non-rectangular shape
    %[X, Y] = meshgrid(linspace(-1, 1, range_bins), linspace(-1, 1, angle_bins));
    %mask = X.^2 + Y.^2 <= 1; % Circular mask
    %map(~mask) = 0; % Apply mask

    % Save map to array
    maps(:, :, i) = map;

    % Save the map, label, and target type into a single .mat file
    filename = sprintf('%s/map_label_%03d.mat', output_folder, i);
    save(filename, 'map', 'labels', 'target_types');
end

% Display counts of each target type
num_absent = sum(target_types == 0);
num_pointlike = sum(target_types == 1);
num_extended = sum(target_types == 2);

fprintf('Number of absent targets: %d\n', num_absent);
fprintf('Number of point-like targets: %d\n', num_pointlike);
fprintf('Number of extended targets: %d\n', num_extended);

% Select one example from each category
idx_absent = find(target_types == 0, 1);
idx_pointlike = find(target_types == 1, 1);
idx_extended = find(target_types == 2, 1);

% Display the maps
figure;

% Absent target
subplot(1,3,1);
imagesc(range, angles, maps(:, :, idx_absent));
colorbar;
xlabel('Range (meters)');
ylabel('Angle of Arrival (degrees)');
title('Absent Target');
axis xy;

% Point-like target
subplot(1,3,2);
imagesc(range, angles, maps(:, :, idx_pointlike));
colorbar;
xlabel('Range (meters)');
ylabel('Angle of Arrival (degrees)');
title('Point-Like Target');
axis xy;

% Extended target
subplot(1,3,3);
imagesc(range, angles, maps(:, :, idx_extended));
colorbar;
xlabel('Range (meters)');
ylabel('Angle of Arrival (degrees)');
title('Extended Target');
axis xy;
