import os
import shutil
import random

# Define dataset split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # 15% validation
TEST_RATIO = 0.15  # 15% test

# Source folders for maps and masks
MAPS_FOLDER = "c:/Users/yigit/Desktop/dataset_final/maps"  # Change this to your maps folder
MASKS_FOLDER = "c:/Users/yigit/Desktop/dataset_final/masks"  # Change this to your masks folder
DEST_FOLDER = "c:/Users/yigit/Desktop/pixelwise_final_2/pixelwise_final"  # Change this to your desired output folder

CATEGORIES = ["no_target", "target"]  # The two categories in your dataset

def split_and_save_data(maps_folder, masks_folder, dest_folder):
    """Splits data into train, val, and test sets and saves them in structured folders."""
    
    # Create destination directories if they don't exist
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_folder, split, "x_data"), exist_ok=True)
        os.makedirs(os.path.join(dest_folder, split, "y_data"), exist_ok=True)
    
    # Process each category separately
    for category in CATEGORIES:
        category_maps_path = os.path.join(maps_folder, category)
        category_masks_path = os.path.join(masks_folder, category)
        
        if not os.path.exists(category_maps_path) or not os.path.exists(category_masks_path):
            print(f"Warning: Category folder '{category}' not found in source.")
            continue
        
        files = [f for f in os.listdir(category_maps_path) if f.endswith(".mat")]
        random.shuffle(files)  # Shuffle to ensure random splitting
        
        # Calculate split indices
        total_files = len(files)
        train_end = int(total_files * TRAIN_RATIO)
        val_end = train_end + int(total_files * VAL_RATIO)

        # Assign files to splits
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Function to copy files to respective folders
        def copy_files(file_list, split):
            for file in file_list:
                map_src = os.path.join(category_maps_path, file)
                mask_src = os.path.join(category_masks_path, file)  # Masks have the same filename
                
                map_dest = os.path.join(dest_folder, split, "x_data", file)
                mask_dest = os.path.join(dest_folder, split, "y_data", file)
                
                if os.path.exists(mask_src):  # Ensure the mask exists before copying
                    shutil.copy2(map_src, map_dest)
                    shutil.copy2(mask_src, mask_dest)
                else:
                    print(f"Warning: Missing mask for {file}, skipping.")

        # Copy files into appropriate directories
        copy_files(train_files, "train")
        copy_files(val_files, "val")
        copy_files(test_files, "test")

        # Print summary
        print(f"Category: {category} -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Run the function
split_and_save_data(MAPS_FOLDER, MASKS_FOLDER, DEST_FOLDER)

print("Data splitting and saving completed successfully!")
