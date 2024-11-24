import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import add_gaussian_noise

def load_and_preprocess_data(data_dir, test_size=0.2, val_size=0.2):
    """
    Load RA map data, normalize, add noise, and split into train/val/test sets.
    
    Parameters:
        data_dir (str): Directory containing RA map files.
        test_size (float): Fraction of data for testing.
        val_size (float): Fraction of training data for validation.

    Returns:
        train_data, val_data, test_data: Preprocessed datasets.
    """
    #To load RA maps from MATLAB, may be modified accordingly!!!!
    ra_maps = [np.load(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]
    
    #Normalize and add Gaussian noise like in the paper but do we need it???? seems unnecessary if this is already what we will do in MATLAB
    normalized_maps = [map_ / np.max(map_) for map_ in ra_maps]
    noisy_maps = [add_gaussian_noise(map_, std_dev=0.05) for map_ in normalized_maps]
    
    #Split data for VALIDATION sets (k-fold)
    train_data, test_data = train_test_split(noisy_maps, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)
    
    #Reshape for CNN input, we need to specify input size
    train_data = np.expand_dims(np.array(train_data), axis=-1)
    val_data = np.expand_dims(np.array(val_data), axis=-1)
    test_data = np.expand_dims(np.array(test_data), axis=-1)
    
    return train_data, val_data, test_data


print("Success")