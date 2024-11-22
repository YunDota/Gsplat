import cv2
import numpy as np
import os
from pathlib import Path

def convert_folder_to_grayscale(source_folder, dest_folder):
    """
    Convert all images in source folder to 3-channel grayscale and save to destination folder.
    
    Args:
        source_folder (str): Path to folder containing source images
        dest_folder (str): Path to folder where grayscale images will be saved
    """
    # Create destination folder if it doesn't exist
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files in source folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(source_folder) 
                  if f.lower().endswith(image_extensions)]
    
    for filename in image_files:
        try:
            # Construct full file paths
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            
            # Read image and convert to grayscale
            img = cv2.imread(source_path)
            if img is None:
                print(f"Could not read {filename}")
                continue
                
            # Convert to grayscale and stack to 3 channels
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_3channel = np.stack([gray] * 3, axis=-1)
            
            # Save the grayscale image
            cv2.imwrite(dest_path, gray_3channel)
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nProcessed {len(image_files)} images")
    print(f"Source folder: {source_folder}")
    print(f"Destination folder: {dest_folder}")

# Example usage
if __name__ == "__main__":
    source_folder = "test_data/bicycle/images"
    dest_folder = "test_data/bicycle/images_2"
    
    convert_folder_to_grayscale(source_folder, dest_folder)