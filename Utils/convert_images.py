import os
from PIL import Image
import argparse
from pathlib import Path

def convert_png_to_jpg(input_folder, output_folder, quality=95):
    """
    Convert all PNG files from input folder to JPG format and save in output folder
    Args:
        input_folder: Path to the folder containing PNG files
        output_folder: Path to save the converted JPG files
        quality: JPEG quality (1-100)
    """
    # Ensure input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Counter for converted files
    count = 0
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if file is PNG
        if filename.lower().endswith('.png'):
            try:
                # Full paths
                png_path = os.path.join(input_folder, filename)
                # Create JPG filename
                jpg_filename = filename[:-4] + '.jpg'
                jpg_path = os.path.join(output_folder, jpg_filename)
                
                # Open PNG image
                with Image.open(png_path) as img:
                    # Convert to RGB mode
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as JPG in output folder
                    img.save(jpg_path, 'JPEG', quality=quality)
                
                count += 1
                print(f"Converted: {filename} -> {jpg_filename}")
                
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")
    
    print(f"\nConversion complete! Converted {count} files.")
    print(f"Converted images saved in: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PNG images to JPG format')
    parser.add_argument('input_folder', help='Path to folder containing PNG images')
    parser.add_argument('output_folder', help='Path to folder where JPG images will be saved')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (1-100)')
    
    args = parser.parse_args()
    
    convert_png_to_jpg(args.input_folder, args.output_folder, args.quality)