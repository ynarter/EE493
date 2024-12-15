import numpy as np
import os
import scipy.io
from sklearn.model_selection import train_test_split
from utils import add_gaussian_noise

def load_and_preprocess_data(data_dir, test_size=0.2, val_size=0.2, noise_std_dev=0.05):
    """
    Load RA map data, normalize, add noise, and split into train/val/test sets for autoencoder training.
    
    Parameters:
        data_dir (str): Directory containing RA map files.
        test_size (float): Fraction of data for testing.
        val_size (float): Fraction of training data for validation.
        noise_std_dev (float): Standard deviation of the Gaussian noise added to the input.

    Returns:
        train_input, train_target, val_input, val_target, test_input, test_target: Processed datasets.
    """
    ra_maps = []

    # Load RA maps
    for file in os.listdir(data_dir):
        if file.endswith(".mat"):  # Process only .mat files
            file_path = os.path.join(data_dir, file)
            mat_data = scipy.io.loadmat(file_path)  # Load .mat file
            ra_map = mat_data['range_angle_map']  # Access the RA map data
            ra_maps.append(ra_map)
    
    # Normalize maps
    ra_maps = [map_ / np.max(map_) for map_ in ra_maps]
    
    # Create noisy maps (inputs for the autoencoder)
    noisy_maps = [add_gaussian_noise(map_, std_dev=noise_std_dev) for map_ in ra_maps]
    
    # Convert to NumPy arrays for easier splitting
    ra_maps = np.array(ra_maps)  # Clean maps (targets)
    noisy_maps = np.array(noisy_maps)  # Noisy maps (inputs)
    
    # Split into training, validation, and test sets
    train_input, test_input, train_target, test_target = train_test_split(
        noisy_maps, ra_maps, test_size=test_size, random_state=42
    )
    train_input, val_input, train_target, val_target = train_test_split(
        train_input, train_target, test_size=val_size, random_state=42
    )
    
    # Reshape for CNN input (add channel dimension)
    train_input = np.expand_dims(train_input, axis=-1)
    train_target = np.expand_dims(train_target, axis=-1)
    val_input = np.expand_dims(val_input, axis=-1)
    val_target = np.expand_dims(val_target, axis=-1)
    test_input = np.expand_dims(test_input, axis=-1)
    test_target = np.expand_dims(test_target, axis=-1)
    
    print(f"Training input shape: {train_input.shape}, Training target shape: {train_target.shape}")
    print(f"Validation input shape: {val_input.shape}, Validation target shape: {val_target.shape}")
    print(f"Test input shape: {test_input.shape}, Test target shape: {test_target.shape}")
    print("Data loaded and preprocessed successfully.")
    
    return train_input, train_target, val_input, val_target, test_input, test_target

# Usage example
"""
data_dir = "path_to_range_angle_maps_folder"  # Replace with your data folder path
train_input, train_target, val_input, val_target, test_input, test_target = load_and_preprocess_data(data_dir)

print(f"Training input shape: {train_input.shape}, Training target shape: {train_target.shape}")
print(f"Validation input shape: {val_input.shape}, Validation target shape: {val_target.shape}")
print(f"Test input shape: {test_input.shape}, Test target shape: {test_target.shape}")
print("Data loaded and preprocessed successfully.")
"""

