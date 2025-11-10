"""
Functions for extracting and managing regions by color for the Arrange Regions step.
"""

from __future__ import annotations

from typing import Dict, Tuple, List
from dataclasses import dataclass
import numpy as np
import cv2 as cv
from PySide6.QtGui import QColor


@dataclass
class RegionData:
    """Data for a single region."""
    image: np.ndarray  # RGBA image of the region
    position: Tuple[int, int]  # Current position (x, y)
    original_position: Tuple[int, int]  # Original position (x, y)
    rotation: float = 0.0  # Rotation angle in degrees
    original_rotation: float = 0.0  # Original rotation angle in degrees


@dataclass
class ColorRegions:
    """Regions grouped by color."""
    color: QColor
    regions: List[RegionData]


def extract_regions_by_color(rgba: np.ndarray, connectivity: int = 8) -> Dict[Tuple[int, int, int], ColorRegions]:
    """
    Extract all disjoint regions from an RGBA image, grouped by color.
    
    Args:
        rgba: RGBA image as numpy array
        connectivity: 4 or 8 for connected components connectivity
        
    Returns:
        Dictionary mapping (R, G, B) tuples to ColorRegions objects
    """
    h, w = rgba.shape[:2]
    
    # Create a mask of non-transparent pixels
    alpha = rgba[:, :, 3]
    non_transparent = alpha > 0
    
    if not np.any(non_transparent):
        return {}
    
    # Get unique colors (only from non-transparent pixels)
    # Reshape to get all pixels
    pixels = rgba[non_transparent].reshape(-1, 4)
    
    # Get unique RGB values (ignore alpha for grouping)
    unique_colors = np.unique(pixels[:, :3], axis=0)
    
    result: Dict[Tuple[int, int, int], ColorRegions] = {}
    
    # For each unique color, find all regions
    for rgb in unique_colors:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        color_key = (r, g, b)
        color = QColor(r, g, b)
        
        # Create mask for this color
        color_mask = np.all(rgba[:, :, :3] == rgb, axis=2) & non_transparent
        
        if not np.any(color_mask):
            continue
        
        # Find connected components for this color
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            color_mask.astype(np.uint8), connectivity=connectivity
        )
        
        regions: List[RegionData] = []
        
        # Extract each region (skip label 0 which is background)
        for label_id in range(1, num_labels):
            # Get bounding box
            x = int(stats[label_id, cv.CC_STAT_LEFT])
            y = int(stats[label_id, cv.CC_STAT_TOP])
            width = int(stats[label_id, cv.CC_STAT_WIDTH])
            height = int(stats[label_id, cv.CC_STAT_HEIGHT])
            
            # Create mask for this specific region
            region_mask = (labels == label_id)
            
            # Extract region image
            region_image = np.zeros((height, width, 4), dtype=np.uint8)
            region_y_start = max(0, y)
            region_y_end = min(h, y + height)
            region_x_start = max(0, x)
            region_x_end = min(w, x + width)
            
            mask_y_start = max(0, -y)
            mask_y_end = mask_y_start + (region_y_end - region_y_start)
            mask_x_start = max(0, -x)
            mask_x_end = mask_x_start + (region_x_end - region_x_start)
            
            region_mask_cropped = region_mask[region_y_start:region_y_end, region_x_start:region_x_end]
            region_image[mask_y_start:mask_y_end, mask_x_start:mask_x_end] = rgba[region_y_start:region_y_end, region_x_start:region_x_end]
            
            # Set transparent pixels outside the region
            region_image[~region_mask_cropped] = [0, 0, 0, 0]
            
            # Create region data
            region_data = RegionData(
                image=region_image,
                position=(x, y),
                original_position=(x, y)
            )
            regions.append(region_data)
        
        if regions:
            result[color_key] = ColorRegions(color=color, regions=regions)
    
    return result

