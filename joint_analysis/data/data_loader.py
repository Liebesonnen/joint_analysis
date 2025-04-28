"""
Data loading functions for joint analysis.
"""

import os
import numpy as np
import torch
from typing import Dict, Optional, List, Tuple, Union


def load_numpy_sequence(file_path: str) -> np.ndarray:
    """
    Load a point cloud sequence from a numpy file.

    Args:
        file_path (str): Path to the numpy file

    Returns:
        ndarray: Point cloud sequence of shape (T, N, 3)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path)

    # Verify data shape
    if len(data.shape) != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected shape (T, N, 3), got {data.shape}")

    return data


def load_pytorch_data(file_path: str, key: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load data from a PyTorch .pt file.

    Args:
        file_path (str): Path to the PyTorch file
        key (str, optional): Key to extract from the data dictionary

    Returns:
        Union[ndarray, Dict[str, ndarray]]: Point cloud data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load PyTorch file
    data_dict = torch.load(file_path, map_location=torch.device('cpu'))

    # Handle the case where we have a data attribute
    if hasattr(data_dict, 'data'):
        data_dict = data_dict.data

    # Extract specific key if provided
    if key is not None:
        if key not in data_dict:
            raise KeyError(f"Key '{key}' not found in data dictionary")

        data = np.array(data_dict[key])

        # Verify data shape
        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(f"Expected shape (T, N, 3), got {data.shape}")

        return data

    # Otherwise, return all data as a dictionary of numpy arrays
    result = {}
    for k, v in data_dict.items():
        result[k] = np.array(v)

    return result


def load_all_sequences_from_directory(directory: str) -> Dict[str, np.ndarray]:
    """
    Load all sequence files from a directory.

    Args:
        directory (str): Directory containing sequence files

    Returns:
        Dict[str, ndarray]: Dictionary mapping file names to point cloud sequences
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    result = {}

    # Walk through the directory and load all .npy files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                name = os.path.splitext(file)[0]
                try:
                    data = load_numpy_sequence(file_path)
                    result[name] = data
                except (ValueError, IOError) as e:
                    print(f"Error loading {file_path}: {e}")

    return result


def subsample_sequence(sequence: np.ndarray, max_points: int = 500) -> np.ndarray:
    """
    Subsample a point cloud sequence to a maximum number of points per frame.

    Args:
        sequence (ndarray): Point cloud sequence of shape (T, N, 3)
        max_points (int): Maximum number of points per frame

    Returns:
        ndarray: Subsampled point cloud sequence of shape (T, min(N, max_points), 3)
    """
    T, N, _ = sequence.shape

    if N <= max_points:
        return sequence

    # Randomly sample points for each frame
    indices = np.random.choice(N, max_points, replace=False)
    return sequence[:, indices, :]


def load_real_datasets() -> Dict[str, np.ndarray]:
    """
    Load real datasets from common paths.
    Note: This function assumes specific paths and might need to be modified based on actual data locations.

    Returns:
        Dict[str, ndarray]: Dictionary mapping dataset names to point cloud sequences
    """
    real_data = {}

    # Define common paths - you may need to modify these based on your actual data locations
    data_paths = [
        {"path": "data/real/drawer/xyz_filtered.pt", "key": "drawer", "name": "Real_Drawer_Data"},
        {"path": "data/real/dishwasher/xyz_filtered.pt", "key": "dishwasher", "name": "Real_Dishwasher_Data"},
        {"path": "data/real/fridge/xyz_filtered.pt", "key": "fridge", "name": "Real_Fridge_Data"}
    ]

    for data_info in data_paths:
        try:
            path = data_info["path"]
            key = data_info["key"]
            name = data_info["name"]

            if os.path.exists(path):
                data = load_pytorch_data(path, key)
                real_data[name] = data
        except Exception as e:
            print(f"Error loading {data_info['name']}: {e}")

    return real_data


def apply_transformation(points: np.ndarray,
                         translation: Optional[np.ndarray] = None,
                         rotation: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply a transformation (translation and/or rotation) to points.

    Args:
        points (ndarray): Points of shape (N, 3)
        translation (ndarray, optional): Translation vector of shape (3,)
        rotation (ndarray, optional): Rotation matrix of shape (3, 3)

    Returns:
        ndarray: Transformed points of shape (N, 3)
    """
    transformed_points = points.copy()

    # Apply rotation if provided
    if rotation is not None:
        transformed_points = transformed_points @ rotation.T

    # Apply translation if provided
    if translation is not None:
        transformed_points = transformed_points + translation

    return transformed_points