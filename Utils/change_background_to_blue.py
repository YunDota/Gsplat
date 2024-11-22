import cv2
import numpy as np
import os
from pathlib import Path

def change_background_to_blue(image, threshold=10, blue_color=(255, 0, 0)):
    """
    Changes background to blue in a 3-channel grayscale image while preserving
    dark pixels that are part of the object.
    
    Parameters:
    image: numpy.ndarray - Input image in BGR format
    threshold (int): Threshold to identify initial dark pixels
    blue_color (tuple): BGR values for blue color
    
    Returns:
    numpy.ndarray: Image with blue background
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Create initial mask of dark pixels
    dark_mask = (image[:,:,0] < threshold).astype(np.uint8) * 255
    
    # Create a seed point mask for flood fill
    seed_points = []
    
    # Check border pixels
    # Top and bottom borders
    for x in range(w):
        if dark_mask[0, x] == 255:
            seed_points.append((x, 0))
        if dark_mask[h-1, x] == 255:
            seed_points.append((x, h-1))
    
    # Left and right borders
    for y in range(h):
        if dark_mask[y, 0] == 255:
            seed_points.append((0, y))
        if dark_mask[y, w-1] == 255:
            seed_points.append((w-1, y))
    
    # Create mask for flood fill
    background_mask = np.zeros((h+2, w+2), np.uint8)
    
    # Flood fill from each seed point
    for seed_x, seed_y in seed_points:
        cv2.floodFill(dark_mask, background_mask, (seed_x, seed_y), 
                      128, flags=4)
    
    # Final background mask is where flood fill reached
    background_mask = (dark_mask == 128).astype(np.uint8) * 255
    
    # Clean up the mask using morphological operations
    kernel = np.ones((3,3), np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    
    # Create output image
    result = image.copy()
    result[background_mask == 255] = blue_color
    
    return result

def process_folder(input_folder, output_folder, threshold=30, blue_color=(255, 0, 0)):
    """
    Process all images in input folder and save results to output folder.
    
    Parameters:
    input_folder (str): Path to folder containing input images
    output_folder (str): Path to folder where processed images will be saved
    threshold (int): Threshold value for dark pixels
    blue_color (tuple): BGR values for blue color
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Process all images in input folder
    total_images = 0
    processed_images = 0
    
    for filename in os.listdir(input_folder):
        # Check if file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            total_images += 1
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Read image
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Failed to read image: {filename}")
                    continue
                
                # Process image
                result = change_background_to_blue(
                    image,
                    threshold=threshold,
                    blue_color=blue_color
                )
                
                # Save result
                cv2.imwrite(output_path, result)
                processed_images += 1
                print(f"Processed: {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing complete!")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed: {total_images - processed_images}")

# Example usage
if __name__ == "__main__":
    # Define folders
    input_folder = "/data/shared/skin/images"    # Replace with your input folder path
    output_folder = "/data/shared/skin/images_3"  # Replace with your output folder path
    
    # Process all images
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        threshold=30,        # Adjust based on your images
        blue_color=(255, 0, 0)  # BGR format
    )