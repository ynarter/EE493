import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import add_gaussian_noise

def load_ra_maps(data_dir, noise_std_dev=0.05):
    ra_map_list = []
    labels_list = []

    # Ensure consistent file ordering
    ra_map_files = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(('.mat', '.npy'))])

    for file in ra_map_files:
        labels = None
        if file.endswith('.mat'):
            mat_data = scipy.io.loadmat(file)
            if 'map' not in mat_data:
                print(f"Warning: No 'map' in {file}. Skipping.")
                continue
            ra_map = mat_data['map']
            labels = mat_data.get('labels', None)  # Use `.get()` for safety
        elif file.endswith('.npy'):
            ra_map = np.load(file)
        else:
            continue

        if ra_map.ndim == 3 and ra_map.shape[-1] == 2:
            magnitude = np.sqrt(ra_map[..., 0]**2 + ra_map[..., 1]**2)
        else:
            magnitude = np.abs(ra_map)

        normalized_map = magnitude / np.max(magnitude)
        ra_map_list.append(normalized_map)
        
        labels_list.append(labels if labels is not None else np.array([]))

    ra_maps = np.array(ra_map_list)
    labels = np.array(labels_list)

    noisy_maps = [add_gaussian_noise(map_, std_dev=noise_std_dev) for map_ in ra_maps]

    print(f"Loaded {len(ra_maps)} RA maps with {len(labels)} labels.")
    return ra_maps, noisy_maps, labels


def split_and_save_datasets(data_dir, save_dir, test_size=0.2, val_size=0.2):
    """
    Load normalized RA maps and labels, split them into train, validation, and test datasets,
    and save them as `.npy` files in respective folders.

    Parameters:
        data_dir (str): Directory containing RA map files.
        save_dir (str): Root directory to save train, validation, and test datasets.
        test_size (float): Fraction of data for testing.
        val_size (float): Fraction of training data for validation.

    Returns:
        None
    """
    # Load all normalized RA maps and labels
    ra_maps, noisy_maps, labels_list = load_ra_maps(data_dir)

    # Split the data into train, test, and validation datasets
    #train_input, test_input, train_target, test_target, train_labels, test_labels = train_test_split(
    #    noisy_maps, ra_maps, labels_list, test_size=test_size, random_state=42
    #)

    #train_input, val_input, train_target, val_target, train_labels, val_labels = train_test_split(
    #    train_input, train_target, train_labels, test_size=val_size, random_state=42
    #)
    
        # Ensure labels_list is a NumPy array before splitting
    combined = list(zip(ra_maps, labels_list))
    train_data, test_data = train_test_split(combined, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)

    # Unpack splits
    train_input, train_labels = zip(*train_data)
    val_input, val_labels = zip(*val_data)
    test_input, test_labels = zip(*test_data)

    # Convert to NumPy arrays
    train_input, val_input, test_input = map(np.array, [train_input, val_input, test_input])
    train_labels, val_labels, test_labels = map(np.array, [train_labels, val_labels, test_labels])
    
    """
    train_input, test_input, train_labels, test_labels = train_test_split(
        ra_maps, labels_list, test_size=test_size, random_state=42
    )

    train_input, val_input, train_labels, val_labels = train_test_split(
        train_input, train_labels, test_size=val_size, random_state=42
    )
    
    """


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

    """
    for dataset, split_name in zip([train_target, val_target, test_target], ["train_target", "val_target", "test_target"]):
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, map_array in enumerate(dataset):
            save_path = os.path.join(split_dir, f"map_{i}.npy")
            np.save(save_path, map_array)
        print(f"Saved {len(dataset)} `.npy` files to {split_dir}")

    """
    
    #os.makedirs(save_dir, exist_ok=True)

    #np.save(os.path.join(save_dir, "train_labels.npy"), train_labels)
    #np.save(os.path.join(save_dir, "val_labels.npy"), val_labels)
    #np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)


    
    for dataset, split_name in zip([train_labels, val_labels, test_labels], ["train_labels", "val_labels", "test_labels"]):
        split_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, labels in enumerate(dataset):
            save_path = os.path.join(split_dir, f"labels_{i}.npy")
            np.save(save_path, labels)
        print(f"Saved {len(dataset)} label `.npy` files to {split_dir}")
    
    print(f"Train labels saved to {os.path.join(save_dir, 'train_labels.npy')} (Shape: {train_labels.shape})")
    print(f"Validation labels saved to {os.path.join(save_dir, 'val_labels.npy')} (Shape: {val_labels.shape})")
    print(f"Test labels saved to {os.path.join(save_dir, 'test_labels.npy')} (Shape: {test_labels.shape})")



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
    return maps

def load_labels(folder_path):
    """
    Load labels from a `.npy` file into a NumPy array.

    Parameters:
        file_path (str): Path to the `.npy` file containing the labels.

    Returns:
        numpy.ndarray: The loaded labels as a NumPy array.
    """
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            label = np.load(file_path)[0]  # Load the .npy file
            labels.append(label)

    labels = np.array(labels)  # Convert list of arrays to a single numpy array
    return labels