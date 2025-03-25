import os
import numpy as np
import scipy.io
#from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#dataset paths
DATASET_PATH = "c:/Users/yigit/Desktop/pixelwise_final_2/pixelwise_final"  # Change this path if needed

#function to load data from a given folder
def load_data_from_folder(x_folder, y_folder):
    """Loads .mat files from given x_data and y_data folders and returns processed data and masks."""
    X_data, y_data_masks, filenames = [], [], []
    
    x_files = sorted(f for f in os.listdir(x_folder) if f.endswith(".mat"))
    y_files = sorted(f for f in os.listdir(y_folder) if f.endswith(".mat"))
    
    for file in x_files:
        x_path = os.path.join(x_folder, file)
        y_path = os.path.join(y_folder, file)  #ground truth mask should have the same filename!!!!!!!!!!!!!
        
        mat_data = scipy.io.loadmat(x_path)
        
        if "range_angle_map_DB" in mat_data:
            matrix = mat_data["range_angle_map_DB"].T  #correct orientation
            matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))  #normalize
            X_data.append(matrix)
            filenames.append(file)
            
            if os.path.exists(y_path):
                y_data = scipy.io.loadmat(y_path)
                
                if "range_angle_map_DB" in y_data:
                    mask_t = y_data["range_angle_map_DB"].astype(np.uint8)  #binary format
                    mask = mask_t.T
                    y_data_masks.append(mask)
                    
                elif "Ground_truth_NN" in y_data:
                    mask_t = y_data["Ground_truth_NN"].astype(np.uint8)  #binary format
                    mask = mask_t.T
                    y_data_masks.append(mask)
                
                else:
                    print(f"Warning: No ground truth mask found in {file}")
                    y_data_masks.append(np.zeros_like(matrix))  #default to all zeros if missing
                    
            else:
                print(f"Warning: Missing mask for {file}")
                y_data_masks.append(np.zeros_like(matrix))  
                
    return np.array(X_data), np.array(y_data_masks), filenames

X_train, y_train_masks, _ = load_data_from_folder(os.path.join(DATASET_PATH, "train/x_data"), os.path.join(DATASET_PATH, "train/y_data"))
X_test, y_test_masks, test_filenames = load_data_from_folder(os.path.join(DATASET_PATH, "test/x_data"), os.path.join(DATASET_PATH, "test/y_data"))
X_val, y_val_masks, _ = load_data_from_folder(os.path.join(DATASET_PATH, "val/x_data"), os.path.join(DATASET_PATH, "val/y_data"))

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_val = X_val[..., np.newaxis]

y_train_masks = y_train_masks[..., np.newaxis] 
y_test_masks = y_test_masks[..., np.newaxis]
y_val_masks = y_val_masks[..., np.newaxis]

#for debugging
print(f"Training Data Shape: {X_train.shape}, Masks Shape: {y_train_masks.shape}")
print(f"Testing Data Shape: {X_test.shape}, Masks Shape: {y_test_masks.shape}")
print(f"Validation Data Shape: {X_val.shape}, Masks Shape: {y_val_masks.shape}")
print(f"Loaded {len(test_filenames)} test filenames.")

# Visualization function
def visualize_samples(X, y, num_samples=5, title="Dataset Samples"):
    """Plots random samples from the dataset along with their ground truth masks."""
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
    fig.suptitle(title, fontsize=16)
    
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(X[idx, :, :, 0], cmap='jet')  # Map visualization
        axes[i, 0].set_title("Range-Angle Map")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(y[idx, :, :, 0], cmap='gray')  # Ground truth mask
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#visualize samples from the training set
#visualize_samples(X_train, y_train_masks, num_samples=5, title="Training Data Samples")
