import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import add_gaussian_noise

def load_ra_maps(data_dir, noise_std_dev=0.05):
    """
    Load, normalize, and return RA map data from `.mat` or `.npy` files as a list of normalized maps.

    Parameters:
        data_dir (str): Directory containing RA map files (`.mat` or `.npy`).

    Returns:
        list: A list of normalized RA maps.
    """
    # List to store normalized RA maps
    ra_map_list = []

    # Get all `.mat` and `.npy` files in the directory
    ra_map_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(('.mat', '.npy'))]

    # Load and normalize RA maps
    for file in ra_map_files:
        # Determine the file format and load the data
        if file.endswith('.mat'):
            mat_data = scipy.io.loadmat(file)
            if 'range_angle_map' not in mat_data:
                continue
            ra_map = mat_data['range_angle_map']
        elif file.endswith('.npy'):
            ra_map = np.load(file)
        else:
            continue

        # If the data has a third dimension (128, 128, 2), compute the magnitude
        if ra_map.ndim == 3 and ra_map.shape[-1] == 2:
            magnitude = np.sqrt(ra_map[..., 0]**2 + ra_map[..., 1]**2)
        else:
            magnitude = np.abs(ra_map)

        # Normalize the RA map
        normalized_map = magnitude / np.max(magnitude)
        ra_map_list.append(normalized_map)

    ra_maps = np.array(ra_map_list)
    print("Size of one map:", ra_maps[1].shape)
    noisy_maps = [add_gaussian_noise(map_, std_dev=noise_std_dev) for map_ in ra_maps]
    
    return ra_maps, noisy_maps

def split_and_save_datasets(data_dir, save_dir, test_size=0.2, val_size=0.2):
    """
    Load normalized RA maps, split them into train, validation, and test datasets,
    and save them as `.npy` files in respective folders.

    Parameters:
        data_dir (str): Directory containing RA map files.
        save_dir (str): Root directory to save train, validation, and test datasets.
        test_size (float): Fraction of data for testing.
        val_size (float): Fraction of training data for validation.

    Returns:
        None
    """
    # Load all normalized RA maps
    ra_maps, noisy_maps = load_ra_maps(data_dir)
    
    train_input, test_input, train_target, test_target = train_test_split(
        noisy_maps, ra_maps, test_size=test_size, random_state=42
    )
    
    train_input, val_input, train_target, val_target = train_test_split(
        train_input, train_target, test_size=val_size, random_state=42
    )
    

    print(f"Train Data Size: {len(train_input)}")
    print(f"Validation Data Size: {len(val_input)}")
    print(f"Test Data Size: {len(test_input)}")

    # Save the datasets as `.npy` files in respective folders
    for dataset, split_name in zip([train_input, val_input, test_input], ["train_input", "val_input", "test_input"]):
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, map_array in enumerate(dataset):
            save_path = os.path.join(split_dir, f"map_{i}.npy")
            np.save(save_path, map_array)
        print(f"Saved {len(dataset)} `.npy` files to {split_dir}")
        
    for dataset, split_name in zip([train_target, val_target, test_target], ["train_target", "val_target", "test_target"]):
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, map_array in enumerate(dataset):
            save_path = os.path.join(split_dir, f"map_{i}.npy")
            np.save(save_path, map_array)
        print(f"Saved {len(dataset)} `.npy` files to {split_dir}")
        

def load_images_from_folder(folder_path, image_size=(256, 256)):
    """
    Load `.npy` files from a folder, normalize them if necessary, and return them as an array.

    Parameters:
        folder_path (str): Path to the folder containing `.npy` files.

    Returns:
        maps: Array containing the loaded maps.
    """
    maps = []
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            map = np.load(file_path)  # Load the .npy file
            map = map / np.max(map)  # Normalize the array to [0, 1]
            maps.append(map)
    
    maps = np.array(maps)  # Convert list of arrays to a single numpy array
    maps = np.expand_dims(maps, axis=-1)
    return maps  # Convert to PyTorch tensor