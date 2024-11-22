#!/usr/bin/env python3
import argparse
import sys
from typing import NamedTuple, Generator, List
from pathlib import Path
import os
import subprocess

class BatchArgs(NamedTuple):
    """Class to store batch processing arguments"""
    source_dir: str
    output_dir: str
    devices: List[int]
    data_factor: int
    mode: str

def parse_batch_args() -> BatchArgs:
    """Parse batch processing arguments and return typed values"""
    parser = argparse.ArgumentParser(
        description='Batch processing for 3D Gaussian Splatting simple_trainer.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '-s', '--source',
        required=True,
        help='Parent directory containing multiple scene folders'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Parent directory for output results'
    )

    parser.add_argument(
        '-d', '--devices',
        required=True,
        help='Comma-separated list of CUDA devices to use (e.g. "0,1,2")'
    )

    parser.add_argument(
        '--data-factor',
        type=int,
        default=1,
        help='Data factor parameter for simple_trainer.py'
    )

    parser.add_argument(
        '--mode',
        default='default',
        choices=['default', 'mcmc'],
        help='Training mode to use'
    )

    args = parser.parse_args()
    
    # Convert devices string to list of integers
    devices = [int(d.strip()) for d in args.devices.split(',')]

    return BatchArgs(
        source_dir=args.source,
        output_dir=args.output,
        devices=devices,
        data_factor=args.data_factor,
        mode=args.mode
    )

def walk_folders(parent_path: str) -> Generator[str, None, None]:
    """
    Walk through a parent folder and yield subfolder paths.
    """
    parent = Path(parent_path).resolve()
    if not parent.is_dir():
        raise ValueError(f"'{parent_path}' is not a valid directory")

    for item in parent.iterdir():
        if item.is_dir():
            yield str(item)

def get_folder_paths(parent_path: str) -> List[str]:
    """Get a list of all subfolder paths."""
    return list(walk_folders(parent_path))

def run_training(scene_folder: str, output_folder: str, devices: List[int], data_factor: int, mode: str) -> None:
    """Run simple_trainer.py with the specified parameters"""
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert devices list to comma-separated string
    devices_str = ','.join(map(str, devices))

    # Construct the command
    cmd = [
        "python",
        "examples/simple_trainer.py",
        mode,
        "--data_dir", scene_folder,
        "--data_factor", str(data_factor),
        "--result_dir", output_folder
    ]

    # Set CUDA devices for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices_str

    # Run the command
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit status {e.returncode}")
        raise

if __name__ == "__main__":
    # Parse the batch processing arguments
    batch_args = parse_batch_args()
    
    # Get all scene folders to process
    scene_folders = get_folder_paths(batch_args.source_dir)
    
    # Process each scene folder
    for scene_folder in scene_folders:
        scene_name = Path(scene_folder).name
        output_folder = str(Path(batch_args.output_dir) / scene_name)
        
        print(f"\nProcessing scene: {scene_name}")
        print(f"Input folder: {scene_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Using CUDA devices: {batch_args.devices}")
        
        try:
            run_training(
                scene_folder=scene_folder,
                output_folder=output_folder,
                devices=batch_args.devices,
                data_factor=batch_args.data_factor,
                mode=batch_args.mode
            )
        except Exception as e:
            print(f"Error processing {scene_name}: {str(e)}")
            continue