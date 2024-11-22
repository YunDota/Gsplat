import imageio.v3 as iio
import imageio
import numpy as np
from pathlib import Path
import tqdm

def process_video(input_path, target_fps=10):
    """
    Read a video file and save it with a modified frame rate using imageio.
    
    Args:
        input_path (str): Path to input video file
        target_fps (int): Desired output frame rate (default: 10)
    """
    try:
        # Create output path
        path = Path(input_path)
        output_path = str(path.parent / f"{path.stem}_modified{path.suffix}")
        
        # Read the video
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data()['fps']
        print(f"Original video FPS: {fps}")
        
        # Create writer with target fps
        writer = imageio.get_writer(output_path, fps=target_fps)
        
        # Process frames
        for frame in tqdm.tqdm(reader, desc="Processing frames"):
            writer.append_data(frame)
            
        # Clean up
        writer.close()
        reader.close()
        
        print(f"Video processed successfully!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        if 'writer' in locals():
            writer.close()
        if 'reader' in locals():
            reader.close()

if __name__ == "__main__":
    input_video = "results/skin/sparse/mcmc/videos/traj_14999.mp4"
    process_video(input_video, target_fps=10)