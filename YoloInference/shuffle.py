import os
import random
import shutil
from tqdm import tqdm

# =============================================================================
# --- CONFIGURATION: USER ACTION REQUIRED ---
# =============================================================================

# 1. Define paths to your source data.
SOURCE_IMAGES_DIR = "/home/ryan/yolov12/C6-5000-5000_2025-06-06-15-15-54 (4)/C6-5000-5000/images"  # <-- All images are here.
SOURCE_LABELS_DIR = "/home/ryan/yolov12/C6-5000-5000_2025-06-06-15-15-54 (4)/C6-5000-5000/output"  # <-- All labels are here.

# 2. Define the path for your new, split dataset.
#    This folder will be created by the script.
OUTPUT_DIR = "/home/ryan/yolov12/DATASETFINALVER"

# 3. Define the split ratio for your validation set.
#    0.2 means 20% of the data will be for validation, 80% for training.
VALIDATION_SPLIT_RATIO = 0.2

# =============================================================================
# --- SCRIPT LOGIC (No changes needed below this line) ---
# =============================================================================

def create_dir_if_not_exists(path):
    """Creates a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)

def move_files(filenames, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    """
    Moves a list of image and their corresponding label files from source to destination.
    It preserves the original image file extension.
    """
    for filename in tqdm(filenames, desc=f"Moving to {os.path.basename(dest_img_dir)}"):
        # Construct the full source path for the image
        source_img_path = os.path.join(source_img_dir, filename)
        
        # Derive the label filename from the image filename
        base_name = os.path.splitext(filename)[0]
        label_filename = f"{base_name}.txt"
        source_lbl_path = os.path.join(source_lbl_dir, label_filename)
        
        # Construct the full destination paths
        dest_img_path = os.path.join(dest_img_dir, filename)
        dest_lbl_path = os.path.join(dest_lbl_dir, label_filename)
        
        # Move the image file
        if os.path.exists(source_img_path):
            shutil.move(source_img_path, dest_img_path)
        
        # Move the corresponding label file, only if it exists
        if os.path.exists(source_lbl_path):
            shutil.move(source_lbl_path, dest_lbl_path)

def main():
    """Main function to execute the dataset split."""
    
    print("--- Starting Dataset Split & Shuffle ---")

    # 1. Create the full directory structure for the output
    print(f"Creating output directory structure at: {OUTPUT_DIR}")
    train_images_dir = os.path.join(OUTPUT_DIR, 'images', 'train')
    train_labels_dir = os.path.join(OUTPUT_DIR, 'labels', 'train')
    val_images_dir = os.path.join(OUTPUT_DIR, 'images', 'val')
    val_labels_dir = os.path.join(OUTPUT_DIR, 'labels', 'val')
    
    for path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        create_dir_if_not_exists(path)
    
    # 2. Get all image filenames from the source directory, filtering for common formats
    try:
        all_filenames = [
            f for f in os.listdir(SOURCE_IMAGES_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    except FileNotFoundError:
        print(f"\nError: Source directory not found at '{SOURCE_IMAGES_DIR}'. Please check the path.")
        return

    if not all_filenames:
        print(f"\nError: No image files found in '{SOURCE_IMAGES_DIR}'. Please check the path.")
        return

    # 3. Shuffle the list of filenames randomly
    print(f"\nFound {len(all_filenames)} images. Shuffling now...")
    random.shuffle(all_filenames)
    
    # 4. Calculate the split index and partition the list
    total_files = len(all_filenames)
    split_index = int(total_files * VALIDATION_SPLIT_RATIO)
    
    val_filenames = all_filenames[:split_index]
    train_filenames = all_filenames[split_index:]
    
    print(f"\nDataset will be split into:")
    print(f"  - Training set:   {len(train_filenames)} images")
    print(f"  - Validation set: {len(val_filenames)} images")
    
    # 5. Move the files to their new training and validation directories
    print("\nStarting file transfer...")
    move_files(train_filenames, SOURCE_IMAGES_DIR, SOURCE_LABELS_DIR, train_images_dir, train_labels_dir)
    move_files(val_filenames, SOURCE_IMAGES_DIR, SOURCE_LABELS_DIR, val_images_dir, val_labels_dir)

    print("\n--- Dataset split complete! ---")
    print(f"Your new, correctly shuffled dataset is ready at: {os.path.abspath(OUTPUT_DIR)}")
    print("\nYou can now update your YAML file to point to these new train/val directories.")


if __name__ == "__main__":
    main()