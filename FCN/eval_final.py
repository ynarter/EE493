import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import weighted_mse
from load import load_data_from_folder
import os
import re
from scipy.ndimage import label, center_of_mass
import scipy.io

# Load your trained model
model = load_model("model_50_32_improved.keras")  # Replace with the correct model path

#Sim maplerinin olduğu directoryler
DATASET_PATH = "c:/Users/yigit/Desktop/pixelwise_final_2/pixelwise_final"
DATASET_PATH_2 = "c:/Users/yigit/Desktop/pixelwise_final_2/pixelwise_final/test/x_data"
X_test, y_test_masks, test_filenames = load_data_from_folder(os.path.join(DATASET_PATH, "test/x_data"), os.path.join(DATASET_PATH, "test/y_data"))

#HW maplerinin olduğu directoryler, sonraki üç satır uncommentlenecek
#DATASET_PATH = "c:/Users/yigit/Desktop/test"
#DATASET_PATH_2 = "c:/Users/yigit/Desktop/test/normal_"
#X_test, y_test_masks, test_filenames = load_data_from_folder(os.path.join(DATASET_PATH, "normal_"), os.path.join(DATASET_PATH, "binary_"))

X_test = X_test[..., np.newaxis]
y_test_masks = y_test_masks[..., np.newaxis]

print(f"Testing Data Shape: {X_test.shape}, Masks Shape: {y_test_masks.shape}")

# Store extracted values
file_info = []

# Regex pattern to extract angle and range
pattern = r"Angle_(-?\d+\.?\d*)Range_(-?\d+\.?\d*)"
#DATASET_PATH_2 = "c:/Users/yigit/Desktop/test/normal_"

# Iterate through all files in the directory
for file in os.listdir(DATASET_PATH_2):
    match = re.search(pattern, file)
    if match and file.endswith(".mat"):
        angle = float(match.group(1))
        range_val = float(match.group(2))
        file_info.append((file, angle, range_val))

# Convert to numpy array for easy processing
file_info = np.array(file_info, dtype=object)

# Extract min/max values
min_angle = file_info[:, 1].min()
max_angle = file_info[:, 1].max()
min_range = file_info[:, 2].min()
max_range = file_info[:, 2].max()

# Find corresponding filenames
min_angle_file = file_info[np.argmin(file_info[:, 1]), 0]
max_angle_file = file_info[np.argmax(file_info[:, 1]), 0]
min_range_file = file_info[np.argmin(file_info[:, 2]), 0]
max_range_file = file_info[np.argmax(file_info[:, 2]), 0]

""" # Print min/max values
print(f"Min Angle: {min_angle}° (File: {min_angle_file})")
print(f"Max Angle: {max_angle}° (File: {max_angle_file})")
print(f"Min Range: {min_range}m (File: {min_range_file})")
print(f"Max Range: {max_range}m (File: {max_range_file})") """

# Function to load .mat files
def load_mat_file(file):
    data = scipy.io.loadmat(os.path.join(DATASET_PATH_2, file))
    # Assuming the map is stored in a key like 'map' in the .mat file
    for key in data:
        if isinstance(data[key], np.ndarray) and data[key].ndim == 2:
            return data[key]
    raise ValueError(f"No valid 2D matrix found in {file}")

# Load maps
min_angle_map = load_mat_file(min_angle_file)
max_angle_map = load_mat_file(max_angle_file)
min_range_map = load_mat_file(min_range_file)
max_range_map = load_mat_file(max_range_file)

# Function to find the max pixel position
def find_max_pixel_position(image):
    return np.unravel_index(np.argmax(image), image.shape)

# Get max pixel positions
min_angle_pos = find_max_pixel_position(min_angle_map)
max_angle_pos = find_max_pixel_position(max_angle_map)
min_range_pos = find_max_pixel_position(min_range_map)
max_range_pos = find_max_pixel_position(max_range_map)

# Row and column positions for Min/Max angles and ranges
row_min_angle = min_angle_pos[0]  # Row corresponding to Min Angle
row_max_angle = max_angle_pos[0]   # Row corresponding to Max Angle
col_min_range = min_range_pos[1]
col_max_range = max_range_pos[1] 

# Initialize separate lists to store angle and range values
true_angles = []
true_ranges = []

# Loop through the test filenames
for filename in test_filenames:
    match = re.search(r"Angle_(-?\d+\.?\d*)Range_(-?\d+\.?\d*)", filename)
    if match:
        angle = float(match.group(1))
        range_val = float(match.group(2))
        true_angles.append(angle)
        true_ranges.append(range_val)
    else:
        # If no match, append 0 for both angle and range
        true_angles.append(0)
        true_ranges.append(0)

# Function to calculate angle for a given row index
def calculate_angle(row_index):
    return min_angle + ((max_angle - min_angle) / (row_max_angle - row_min_angle)) * (row_index - row_min_angle)

# Function to calculate range for a given column index
def calculate_range(col_index):
    return min_range + ((max_range - min_range) / (col_max_range - col_min_range)) * (col_index - col_min_range)

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions to binary (if necessary, based on the sigmoid output)
y_pred_bin = (y_pred > 0.9).astype(np.uint8)

# Store center positions and corresponding masks
center_positions = []
center_masks = np.zeros_like(y_pred_bin)  # Create empty masks of the same shape


for idx in range(len(y_pred_bin)):
    prob_map = y_pred[idx].squeeze()  # Use probability values

    # Check if the entire map is zero
    if np.all(y_pred_bin[idx] == 0):
        center_positions.append([])  # No centers
        continue  # Skip to next iteration

    # Find the coordinates of the pixel with the highest probability
    max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)  # Get (row, col) coordinates
    center_positions.append([max_idx])  # Store as a list with one tuple
    
    # Create a new mask with only the computed center pixel set to 1
    center_masks[idx, max_idx[0], max_idx[1]] = 1

""" # Print stored center positions
for i, centers in enumerate(center_positions):
    print(f"Sample {i+1} center positions: {centers}") """
    
# Initialize lists to store estimated angles and ranges for each sample
estimated_angles = []
estimated_ranges = []

# Loop over each sample in center_positions
for centers in center_positions:
    sample_angles = []
    sample_ranges = []
    
    # If there are centers, calculate the angle and range for each
    if centers:
        for center in centers:
            col_idx, row_idx = center
            # Calculate angle and range
            angle = calculate_angle(row_idx)
            range_val = calculate_range(col_idx)
            sample_angles.append(angle)
            sample_ranges.append(range_val)
    else:
        # If no centers, add zero for both angle and range
        sample_angles.append(0)
        sample_ranges.append(0)
    
    # Append the results for this sample to the lists
    estimated_angles.append(sample_angles)
    estimated_ranges.append(sample_ranges)

""" # Display the results
for idx in range(len(estimated_angles)):
    print(f"Sample {idx + 1}:")
    print(f"Estimated Angles: {estimated_angles[idx]}")
    print(f"Estimated Ranges: {estimated_ranges[idx]}") """

""" # Plot a few examples for visualization
num_samples_to_plot = max(5, len(y_pred_bin))  # Plot up to 5 examples
plt.figure(figsize=(15, 10))

for i in range(num_samples_to_plot):
    plt.subplot(num_samples_to_plot, 2, 2*i + 1)
    plt.imshow(y_pred_bin[i].squeeze().T, cmap='gray')
    plt.title(f"Predicted Map - Sample {i+1}")
    plt.axis('off')

    plt.subplot(num_samples_to_plot, 2, 2*i + 2)
    plt.imshow(center_masks[i].squeeze().T, cmap='gray')
    plt.title(f"Center Pixel Mask - Sample {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show() """

# Flatten the arrays for metric calculations
y_test_flat = y_test_masks.flatten()
y_pred_flat = y_pred_bin.flatten()

# Calculate metrics
accuracy = accuracy_score(y_test_flat, y_pred_flat)
precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

# Print evaluation metrics as percentages with 3 digits
print(f"Accuracy: {accuracy * 100:.3f}%")
print(f"Precision: {precision * 100:.3f}%")
print(f"Recall: {recall * 100:.3f}%")
print(f"F1 Score: {f1 * 100:.3f}%")

# Number of samples to plot
num_samples_to_plot = 5

# Randomly select indices
random_indices = np.random.choice(len(X_test), size=num_samples_to_plot, replace=False)

for i, idx in enumerate(random_indices):
    angle = true_angles[idx]
    range_val = true_ranges[idx]
    title_suffix = f"Angle: {angle}°, Range: {range_val}m"
    
    # Input map
    plt.subplot(num_samples_to_plot, 3, 3*i + 1)
    plt.imshow(X_test[idx].squeeze().T, cmap='jet')
    plt.title(f"Sample {i+1} - Input Map\n{title_suffix}")
    plt.axis('off')
    
    # Ground truth
    plt.subplot(num_samples_to_plot, 3, 3*i + 2)
    plt.imshow(y_test_masks[idx].squeeze().T, cmap='gray')
    plt.title(f"Ground Truth - Sample {i+1}\n{title_suffix}")
    plt.axis('off')
    
    # Predicted map
    plt.subplot(num_samples_to_plot, 3, 3*i + 3)
    plt.imshow(y_pred_bin[idx].squeeze().T, cmap='gray')
    plt.title(f"Predicted Map - Sample {i+1}\n{title_suffix}")
    plt.axis('off')

plt.tight_layout()
plt.show()

for i, idx in enumerate(random_indices):
    angle = true_angles[idx]
    range_val = true_ranges[idx]
    title_suffix = f"Angle: {angle}°, Range: {range_val}m"
    
    # Input map
    plt.subplot(num_samples_to_plot, 2, 2*i + 1)
    plt.imshow(X_test[idx].squeeze().T, cmap='jet')
    plt.title(f"Sample {i+1} - Input Map\n{title_suffix}")
    plt.axis('off')
    
    # Ground truth
    plt.subplot(num_samples_to_plot, 2, 2*i + 2)
    plt.imshow(center_masks[idx].squeeze().T, cmap='gray')
    plt.title(f"Sample {i+1} - Input Map\n{title_suffix}")
    plt.axis('off')


plt.tight_layout()
plt.show()

# Plotting True vs Estimated Angles
plt.figure(figsize=(10, 6))
plt.plot(true_angles, label="True Angles", marker='o', linestyle='-', color='blue')
plt.plot(estimated_angles, label="Estimated Angles", marker='x', linestyle='--', color='red')
plt.title("True vs Estimated Angles")
plt.xlabel("Sample Index")
plt.ylabel("Angle (°)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Plotting True vs Estimated Ranges
plt.figure(figsize=(10, 6))
plt.plot(true_ranges, label="True Ranges", marker='o', linestyle='-', color='blue')
plt.plot(estimated_ranges, label="Estimated Ranges", marker='x', linestyle='--', color='red')
plt.title("True vs Estimated Ranges")
plt.xlabel("Sample Index")
plt.ylabel("Range (m)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Calculate R² score for angles using sklearn's r2_score function
r2_angle = r2_score(true_angles, estimated_angles)

# Calculate RMSE for angles
rmse_angle = np.sqrt(mean_squared_error(true_angles, estimated_angles))

# Calculate R² score for ranges using sklearn's r2_score function
r2_range = r2_score(true_ranges, estimated_ranges)

# Calculate RMSE for ranges
rmse_range = np.sqrt(mean_squared_error(true_ranges, estimated_ranges))

# Report R² score as a percentage
r2_angle_percentage = r2_angle * 100
r2_range_percentage = r2_range * 100

# Print the results
print(f"R² Score for Angles: {r2_angle_percentage:.2f}%")
print(f"RMSE for Angles: {rmse_angle:.4f}")
print(f"R² Score for Ranges: {r2_range_percentage:.2f}%")
print(f"RMSE for Ranges: {rmse_range:.4f}")


""" # Metrics
metric_names = ["R² Score (%)", "RMSE", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
angle_values = [r2_angle * 100, rmse_angle, accuracy * 100, precision * 100, recall * 100, f1 * 100]
range_values = [r2_range * 100, rmse_range, np.nan, np.nan, np.nan, np.nan]  # Replace None with np.nan

# Bar chart settings
x = np.arange(len(metric_names))
width = 0.3  # Bar width

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, angle_values, width, label="Angle", color="#1f77b4")
bars2 = ax.bar(x + width/2, np.nan_to_num(range_values, nan=0), width, label="Range", color="#ff7f0e", alpha=0.8)

# Add value labels
for bars, values in zip([bars1, bars2], [angle_values, range_values]):
    for bar, value in zip(bars, values):
        if not np.isnan(value):  # Avoid labeling NaN bars
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Labels and aesthetics
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold', rotation=20)
ax.set_ylabel("Value", fontsize=12, fontweight='bold')
ax.set_title("Performance Metrics for Angle and Range Estimation", fontsize=13, fontweight='bold')
ax.legend()

# Display the plot
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

x = np.arange(len(metric_names))
width = 0.4  # Bar width

# Define different colors for each column
angle_colors = ["#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c"]  # Different shades for angle
range_colors = ["#ff7f0e", "#d62728", "#9467bd", "#8c564b"]  # Different shades for range

fig, ax = plt.subplots(figsize=(10, 5))

bars1 = [ax.bar(x[i] - width/2, angle_values[i], width, color=angle_colors[i], label="Angle" if i == 0 else "")
         for i in range(len(metric_names))]
bars2 = [ax.bar(x[i] + width/2, np.nan_to_num(range_values[i], nan=0), width, color=range_colors[i], alpha=0.8, label="Range" if i == 0 else "")
         for i in range(len(metric_names))]

# Add value labels
for bars, values in zip([bars1, bars2], [angle_values, range_values]):
    for bar, value in zip(bars, values):
        ax.text(bar[0].get_x() + bar[0].get_width()/2, bar[0].get_height(), f"{value:.2f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Labels and aesthetics
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold', rotation=20)
ax.set_ylabel("Value", fontsize=12, fontweight='bold')
ax.set_title("Performance Metrics for Angle and Range Estimation", fontsize=13, fontweight='bold')
ax.legend()

# Display the plot
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show() """

# Metrics for Range
metric_names = ["Classification Accuracy (%)", "Range R² Score (%)", "Range RMSE", "Angle R² Score (%)", "Angle RMSE", "Precision (%)", "Recall (%)", "F1 Score (%)"]
range_values = [accuracy * 100, r2_range * 100, rmse_range, r2_angle * 100, rmse_angle, precision * 100, recall * 100, f1 * 100]

# Bar chart settings
x = np.arange(len(metric_names))
width = 0.3  # Reduced bar width for better spacing

# Define different colors for each bar
colors = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f"   # Gray
]


fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(x, range_values, width, color=colors, alpha=0.8)

# Add value labels above bars
for bar, value in zip(bars, range_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}", 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Labels and aesthetics
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold', rotation=15)
ax.set_ylabel("Value", fontsize=12, fontweight='bold')
ax.set_title("Performance Metrics", fontsize=13, fontweight='bold')

# Grid for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display the plot
plt.show()