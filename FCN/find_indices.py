import os
import re
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Define the dataset path
DATASET_PATH = "c:/Users/yigit/Desktop/pixelwise_final/test/x_data"

# Regex pattern to extract angle and range
pattern = r"Angle_(-?\d+\.?\d*)Range_(-?\d+\.?\d*)"

# Store file information
file_info = []
angle_first_file = {}  # Dictionary to store first encountered file per angle

# Iterate through all files in the directory
for file in os.listdir(DATASET_PATH):
    match = re.search(pattern, file)
    if match and file.endswith(".mat"):
        angle = float(match.group(1))
        range_val = float(match.group(2))
        
        # Store the first encountered file for each unique angle
        if angle not in angle_first_file:
            angle_first_file[angle] = file

# Convert angles to sorted list
unique_angles = sorted(angle_first_file.keys())

print("Unique Angles and Corresponding Files:")
for angle in unique_angles:
    print(f"Angle {angle}°: File {angle_first_file[angle]}")

# Function to load .mat files
def load_mat_file(file):
    data = scipy.io.loadmat(os.path.join(DATASET_PATH, file))
    for key in data:
        if isinstance(data[key], np.ndarray) and data[key].ndim == 2:
            return data[key]
    raise ValueError(f"No valid 2D matrix found in {file}")

# Function to find the max pixel position
def find_max_pixel_position(image):
    return np.unravel_index(np.argmax(image), image.shape)

# Load and process one map for each unique angle
for angle in unique_angles:
    file = angle_first_file[angle]
    print(f"Processing Angle: {angle}° - File: {file}")
    angle_map = load_mat_file(file)
    max_pixel_pos = find_max_pixel_position(angle_map)
    print(f"Max Pixel Position: {max_pixel_pos}")
    
    # Plot the map
    plt.figure(figsize=(5, 5))
    plt.imshow(angle_map, cmap="jet")
    plt.scatter(max_pixel_pos[1], max_pixel_pos[0], color="red", marker="x", s=100, label="Max Pixel")
    plt.title(f"Angle {angle}° - File: {file}")
    plt.legend()
    plt.axis("off")
    plt.show()