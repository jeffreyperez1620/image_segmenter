#!/usr/bin/env python3
"""
Utilities for transparency handling and alpha channel manipulation.
"""

import numpy as np


def clamp_alpha_channel(rgba: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Clamp alpha channel to binary values (0 or 255) based on threshold.
    
    Args:
        rgba: Input RGBA image as numpy array
        threshold: Alpha threshold for binary conversion (default: 128)
        
    Returns:
        RGBA image with clamped alpha channel
    """
    result = rgba.copy()
    alpha = result[:, :, 3]
    
    # Convert alpha to binary: 0 if below threshold, 255 if above
    binary_alpha = np.where(alpha >= threshold, 255, 0)
    result[:, :, 3] = binary_alpha
    
    return result


def normalize_alpha_channel(rgba: np.ndarray) -> np.ndarray:
    """
    Ensure alpha=255 for all non-transparent pixels.
    
    Args:
        rgba: Input RGBA image as numpy array
        
    Returns:
        RGBA image with normalized alpha channel
    """
    result = rgba.copy()
    alpha = result[:, :, 3]
    
    # Set alpha to 255 for all non-transparent pixels
    non_transparent = alpha > 0
    result[non_transparent, 3] = 255
    
    return result


def has_transparency_issues(rgba: np.ndarray) -> bool:
    """
    Check if image has transparency issues (non-binary alpha values).
    
    Args:
        rgba: Input RGBA image as numpy array
        
    Returns:
        True if transparency issues detected, False otherwise
    """
    alpha = rgba[:, :, 3]
    non_transparent = alpha > 0
    
    if not np.any(non_transparent):
        return False
    
    # Check if any non-transparent pixels have alpha != 255
    non_binary_alpha = np.any((alpha[non_transparent] != 255) & (alpha[non_transparent] != 0))
    return non_binary_alpha

