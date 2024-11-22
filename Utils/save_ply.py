import numpy as np
from plyfile import PlyData, PlyElement
import torch
import os

def save_ply(gaussian_data, filename, device="cuda"):
    """
    Save 3D Gaussian model data to PLY file matching the original format
    
    Args:
        gaussian_data: Dictionary containing Gaussian parameters
        filename: Output PLY file path
        device: Computation device
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Extract and prepare data
    xyz = gaussian_data['means'].detach().cpu().numpy()
    normals = np.zeros_like(xyz)  # Empty normals
    
    # Features DC and rest need to be properly reshaped
    f_dc = gaussian_data['sh0'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussian_data['shN'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    
    # Other properties - ensure correct dimensions
    opacities = gaussian_data['opacities'].detach().cpu().numpy()
    if opacities.ndim == 1:
        opacities = opacities.reshape(-1, 1)
        
    scales = gaussian_data['scales'].detach().cpu().numpy()
    rotations = gaussian_data['quats'].detach().cpu().numpy()

    # Construct attribute list
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    
    # Add feature DC terms
    for i in range(f_dc.shape[1]):
        attributes.append(f'f_dc_{i}')
    
    # Add feature rest terms
    for i in range(f_rest.shape[1]):
        attributes.append(f'f_rest_{i}')
    
    # Add opacity
    attributes.append('opacity')
    
    # Add scales
    for i in range(scales.shape[1]):
        attributes.append(f'scale_{i}')
    
    # Add rotations
    for i in range(rotations.shape[1]):
        attributes.append(f'rot_{i}')

    # Create dtype for structured array
    dtype_full = [(attribute, 'f4') for attribute in attributes]
    
    # Create empty array with the defined dtype
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    # Ensure all arrays have correct shape before concatenation
    if f_dc.ndim == 1:
        f_dc = f_dc.reshape(-1, 1)
    if f_rest.ndim == 1:
        f_rest = f_rest.reshape(-1, 1)
    
    # Print shapes for debugging
    # print(f"Shapes - xyz: {xyz.shape}, normals: {normals.shape}, f_dc: {f_dc.shape}, "
    #       f"f_rest: {f_rest.shape}, opacities: {opacities.shape}, scales: {scales.shape}, "
    #       f"rotations: {rotations.shape}")
    
    # Concatenate all attributes in the correct order
    attributes = np.concatenate((
        xyz,        # positions [N, 3]
        normals,    # empty normals [N, 3]
        f_dc,       # DC features [N, DC]
        f_rest,     # rest features [N, REST]
        opacities,  # opacity [N, 1]
        scales,     # scales [N, 3]
        rotations   # rotations [N, 4]
    ), axis=1)
    
    # Fill the structured array
    elements[:] = list(map(tuple, attributes))
    
    # Create PLY element and save
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)
    print(f"Saved Gaussian model to: {filename}")