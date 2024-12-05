% Parameters
range_resolution = 0.5; % Range resolution (example value in meters)
angle_resolution = 0.1; % Angle resolution (example value in radians)

% Define frequency scaling factors for sinc functions
range_bandwidth = 1 / range_resolution; % Bandwidth for range
angle_bandwidth = 1 / angle_resolution; % Bandwidth for angle

% Generate range and angle grids
range_bins = 256; % Number of range bins
angle_bins = 256; % Number of angle bins
range = linspace(-10, 10, range_bins); % Range values (example limits)
angles = linspace(-pi/2, pi/2, angle_bins); % Angle values (in radians)

% Create a meshgrid for range and angle
[Range, Angle] = meshgrid(range, angles);

% Initialize parameters for dataset generation
num_maps = 500; % Number of maps to generate
output_folder = 'range_angle_maps'; % Folder to store individual maps

% Create the output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Generate and save maps with peaks at different locations
for i = 1:num_maps
    % Randomly select a peak position for range and angle
    peak_range = -8 + (8 - (-8)) * rand; % Random peak within range limits
    peak_angle = -pi/4 + (pi/4 - (-pi/4)) * rand; % Random peak within angle limits
    
    % Shift the sinc function to peak at the specified range and angle
    range_shifted = Range - peak_range; % Shift the range grid
    angle_shifted = Angle - peak_angle; % Shift the angle grid
    
    % Generate a range-angle map with the shifted sinc functions
    range_angle_map = sinc(range_shifted * range_bandwidth) .* sinc(sin(angle_shifted) * angle_bandwidth);
    
    % Normalize the map for consistency
    range_angle_map = range_angle_map / max(range_angle_map(:));
    
    % Save the map to a file
    file_name = fullfile(output_folder, sprintf('range_angle_map_%03d.mat', i));
    save(file_name, 'range_angle_map');
end

% Display completion message
fprintf('All maps have been saved to the folder: %s\n', output_folder);

% Plot a random sample map for verification
figure;
sample_index = randi(num_maps); % Select a random sample
sample_file = fullfile(output_folder, sprintf('range_angle_map_%03d.mat', sample_index));
load(sample_file, 'range_angle_map'); % Load the map
imagesc(range, angles, range_angle_map); % Visualize as an image
colorbar;
xlabel('Range (m)');
ylabel('Angle (radians)');
title(sprintf('Sample Range-Angle Map (Index: %d)', sample_index));
axis xy; % Flip y-axis to match conventional orientation
