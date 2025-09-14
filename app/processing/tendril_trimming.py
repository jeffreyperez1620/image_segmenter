"""
Tendril Trimming Algorithm v2

This module contains the core algorithm for detecting and removing tendrils from images.
A tendril is defined as a pixel where either its horizontal OR vertical thickness
is less than or equal to a user-defined threshold.

The algorithm works by:
1. Using alpha channel as sentinel values to mark different types of tendrils
2. Processing horizontal and vertical tendrils in separate passes
3. Collecting changes and applying them together for better merging
4. Iterating until no more tendrils are found

Author: AI Assistant
Date: 2024
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class TendrilTrimmer:
    """Main class for tendril trimming operations using alpha sentinel values."""
    
    def __init__(self):
        """Initialize the tendril trimmer."""
        # Sentinel values for alpha channel
        self.NORMAL_PIXEL = 10      # Not a tendril
        self.HORIZONTAL_THIN = 11   # Horizontally thin
        self.VERTICAL_THIN = 12     # Vertically thin
        self.BOTH_THIN = 13         # Both horizontally and vertically thin
    
    def trim_tendrils(self, rgba: np.ndarray, threshold: int, max_iterations: int = 30) -> Tuple[np.ndarray, int, str]:
        """
        Trim tendrils from an image using the new alpha sentinel approach.
        
        Args:
            rgba: Input image as RGBA numpy array
            threshold: Maximum thickness for a pixel to be considered a tendril
            max_iterations: Maximum number of iterations to perform
            
        Returns:
            Tuple of (processed_image, iterations_used, status_message)
        """
        # Create a copy to avoid modifying the original
        result = rgba.copy()
        
        # Check if image has any non-transparent pixels
        alpha = result[:, :, 3]
        if not np.any(alpha > 0):
            return result, 0, "No non-transparent pixels found"
        
        iteration = 0
        
        while iteration < max_iterations:
            # 1. Mark tendrils and count them
            tendril_count = self._mark_tendrils(result, threshold)
            
            # 2. Early exit if no tendrils found
            if tendril_count == 0:
                break
                
            # 3. Process tendrils (collect changes, then apply)
            self._process_tendrils(result, threshold)
            
            iteration += 1
        
        # 4. Restore alpha channel to full opacity
        self._restore_alpha_channel(result, rgba)
        
        return result, iteration, f"Completed after {iteration} iterations"
    
    def _mark_tendrils(self, rgba: np.ndarray, threshold: int) -> int:
        """
        Mark tendrils using alpha sentinel values.
        
        Args:
            rgba: Input image as RGBA numpy array
            threshold: Maximum thickness for a pixel to be considered a tendril
            
        Returns:
            Number of tendril pixels found
        """
        height, width = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        
        # Convert RGB to single integer representation for efficient comparison
        rgb_int = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2].astype(np.uint32)
        
        tendril_count = 0
        
        # First, set all non-transparent pixels to NORMAL_PIXEL
        non_transparent = alpha > 0
        alpha[non_transparent] = self.NORMAL_PIXEL
        
        # For each pixel, check if it's a tendril
        for y in range(height):
            for x in range(width):
                if not non_transparent[y, x]:
                    continue
                
                current_color = rgb_int[y, x]
                is_horizontal_thin = False
                is_vertical_thin = False
                
                # Check horizontal thickness
                horizontal_thickness = self._calculate_horizontal_thickness(rgb_int, alpha, x, y, current_color, width)
                if horizontal_thickness <= threshold:
                    is_horizontal_thin = True
                
                # Check vertical thickness
                vertical_thickness = self._calculate_vertical_thickness(rgb_int, alpha, x, y, current_color, height)
                if vertical_thickness <= threshold:
                    is_vertical_thin = True
                
                # Mark pixel based on tendril type
                if is_horizontal_thin and is_vertical_thin:
                    alpha[y, x] = self.BOTH_THIN
                    tendril_count += 1
                elif is_horizontal_thin:
                    alpha[y, x] = self.HORIZONTAL_THIN
                    tendril_count += 1
                elif is_vertical_thin:
                    alpha[y, x] = self.VERTICAL_THIN
                    tendril_count += 1
        
        return tendril_count
    
    def _calculate_horizontal_thickness(self, rgb_int: np.ndarray, alpha: np.ndarray, x: int, y: int, current_color: int, width: int) -> int:
        """Calculate horizontal thickness of a pixel."""
        left_dist = 0
        right_dist = 0
        
        # Count distance to left
        for dx in range(1, x + 1):
            if x - dx < 0 or alpha[y, x - dx] == 0 or rgb_int[y, x - dx] != current_color:
                break
            left_dist += 1
        
        # Count distance to right
        for dx in range(1, width - x):
            if x + dx >= width or alpha[y, x + dx] == 0 or rgb_int[y, x + dx] != current_color:
                break
            right_dist += 1
        
        return left_dist + right_dist + 1
    
    def _calculate_vertical_thickness(self, rgb_int: np.ndarray, alpha: np.ndarray, x: int, y: int, current_color: int, height: int) -> int:
        """Calculate vertical thickness of a pixel."""
        up_dist = 0
        down_dist = 0
        
        # Count distance up
        for dy in range(1, y + 1):
            if y - dy < 0 or alpha[y - dy, x] == 0 or rgb_int[y - dy, x] != current_color:
                break
            up_dist += 1
        
        # Count distance down
        for dy in range(1, height - y):
            if y + dy >= height or alpha[y + dy, x] == 0 or rgb_int[y + dy, x] != current_color:
                break
            down_dist += 1
        
        return up_dist + down_dist + 1
    
    def _process_tendrils(self, rgba: np.ndarray, threshold: int) -> None:
        """
        Process all tendrils by collecting changes and applying them together.
        
        Args:
            rgba: Input image as RGBA numpy array
            threshold: Maximum thickness for a pixel to be considered a tendril
        """
        # Process horizontal tendrils first (including BOTH_THIN pixels)
        self._process_horizontal_tendrils(rgba, threshold)
        
        # Then process vertical tendrils (only VERTICAL_THIN pixels)
        self._process_vertical_tendrils(rgba, threshold)
    
    def _process_horizontal_tendrils(self, rgba: np.ndarray, threshold: int) -> None:
        """Process horizontal tendrils by scanning horizontally."""
        height, width = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        
        # Collect all changes before applying them
        changes = {}  # (y, x) -> new_color
        
        for y in range(height):
            x = 0
            while x < width:
                if alpha[y, x] in [self.HORIZONTAL_THIN, self.BOTH_THIN]:
                    # Found start of horizontal tendril
                    scan_line_info = self._scan_horizontal_line(rgba, x, y, threshold)
                    if scan_line_info:
                        start_x, end_x, majority_color = scan_line_info
                        
                        if end_x - start_x + 1 > threshold:
                            # Scan line is long enough - set all pixels to majority color
                            for px in range(start_x, end_x + 1):
                                if alpha[y, px] in [self.HORIZONTAL_THIN, self.BOTH_THIN]:
                                    changes[(y, px)] = majority_color
                        else:
                            # Scan line is too short - use adjacent non-tendril colors
                            for px in range(start_x, end_x + 1):
                                if alpha[y, px] in [self.HORIZONTAL_THIN, self.BOTH_THIN]:
                                    new_color = self._find_adjacent_color(rgba, px, y)
                                    if new_color is not None:
                                        changes[(y, px)] = new_color
                        
                        x = end_x + 1  # Skip to end of scan line
                    else:
                        x += 1
                else:
                    x += 1
        
        # Apply all changes
        for (y, x), new_color in changes.items():
            rgb[y, x] = new_color
            alpha[y, x] = self.NORMAL_PIXEL  # Mark as processed
    
    def _process_vertical_tendrils(self, rgba: np.ndarray, threshold: int) -> None:
        """Process vertical tendrils by scanning vertically."""
        height, width = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        
        # Collect all changes before applying them
        changes = {}  # (y, x) -> new_color
        
        for x in range(width):
            y = 0
            while y < height:
                if alpha[y, x] == self.VERTICAL_THIN:
                    # Found start of vertical tendril
                    scan_line_info = self._scan_vertical_line(rgba, x, y, threshold)
                    if scan_line_info:
                        start_y, end_y, majority_color = scan_line_info
                        
                        if end_y - start_y + 1 > threshold:
                            # Scan line is long enough - set all pixels to majority color
                            for py in range(start_y, end_y + 1):
                                if alpha[py, x] == self.VERTICAL_THIN:
                                    changes[(py, x)] = majority_color
                        else:
                            # Scan line is too short - use adjacent non-tendril colors
                            for py in range(start_y, end_y + 1):
                                if alpha[py, x] == self.VERTICAL_THIN:
                                    new_color = self._find_adjacent_color(rgba, x, py)
                                    if new_color is not None:
                                        changes[(py, x)] = new_color
                        
                        y = end_y + 1  # Skip to end of scan line
                    else:
                        y += 1
                else:
                    y += 1
        
        # Apply all changes
        for (y, x), new_color in changes.items():
            rgb[y, x] = new_color
            alpha[y, x] = self.NORMAL_PIXEL  # Mark as processed
    
    def _scan_horizontal_line(self, rgba: np.ndarray, start_x: int, y: int, threshold: int) -> Optional[Tuple[int, int, np.ndarray]]:
        """Scan horizontally to find the extent of a tendril line and determine majority color."""
        width = rgba.shape[1]
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        
        # Find the extent of the tendril line
        end_x = start_x
        while end_x + 1 < width and alpha[y, end_x + 1] in [self.HORIZONTAL_THIN, self.BOTH_THIN]:
            end_x += 1
        
        # Count colors in the scan line
        color_counts = {}
        for x in range(start_x, end_x + 1):
            if alpha[y, x] in [self.HORIZONTAL_THIN, self.BOTH_THIN]:
                color_key = tuple(rgb[y, x])
                color_counts[color_key] = color_counts.get(color_key, 0) + 1
        
        if not color_counts:
            return None
        
        # Find majority color (use leftmost in case of tie)
        majority_color = max(color_counts, key=lambda k: (color_counts[k], -list(color_counts.keys()).index(k)))
        
        return start_x, end_x, np.array(majority_color, dtype=np.uint8)
    
    def _scan_vertical_line(self, rgba: np.ndarray, x: int, start_y: int, threshold: int) -> Optional[Tuple[int, int, np.ndarray]]:
        """Scan vertically to find the extent of a tendril line and determine majority color."""
        height = rgba.shape[0]
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        
        # Find the extent of the tendril line
        end_y = start_y
        while end_y + 1 < height and alpha[end_y + 1, x] == self.VERTICAL_THIN:
            end_y += 1
        
        # Count colors in the scan line
        color_counts = {}
        for y in range(start_y, end_y + 1):
            if alpha[y, x] == self.VERTICAL_THIN:
                color_key = tuple(rgb[y, x])
                color_counts[color_key] = color_counts.get(color_key, 0) + 1
        
        if not color_counts:
            return None
        
        # Find majority color (use topmost in case of tie)
        majority_color = max(color_counts, key=lambda k: (color_counts[k], -list(color_counts.keys()).index(k)))
        
        return start_y, end_y, np.array(majority_color, dtype=np.uint8)
    
    def _find_adjacent_color(self, rgba: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
        """Find a non-tendril color adjacent to the given pixel."""
        height, width = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        
        # Check 4-neighbors
        neighbors = [
            (y-1, x),   # up
            (y+1, x),   # down
            (y, x-1),   # left
            (y, x+1)    # right
        ]
        
        for ny, nx in neighbors:
            if (0 <= ny < height and 0 <= nx < width and 
                alpha[ny, nx] == self.NORMAL_PIXEL):
                return rgb[ny, nx].copy()
        
        return None
    
    def _restore_alpha_channel(self, result: np.ndarray, original: np.ndarray) -> None:
        """Restore alpha channel to full opacity for all non-transparent pixels."""
        # Set alpha to 255 for all pixels that were originally non-transparent
        original_non_transparent = original[:, :, 3] > 0
        result[original_non_transparent, 3] = 255
    
    # Backward compatibility methods for the test app
    def _trim_tendrils_in_iteration(self, rgba: np.ndarray, threshold: int) -> int:
        """
        Backward compatibility method for the test app.
        This method runs one iteration of the new algorithm and returns the number of pixels changed.
        """
        # Mark tendrils
        tendril_count = self._mark_tendrils(rgba, threshold)
        
        if tendril_count == 0:
            return 0
        
        # Process tendrils
        self._process_tendrils(rgba, threshold)
        
        return tendril_count
    
    def _apply_color_selection_to_magenta(self, rgba: np.ndarray) -> int:
        """
        Backward compatibility method for the test app.
        This method processes any remaining tendril pixels and returns the number of pixels recolored.
        """
        height, width = rgba.shape[:2]
        alpha = rgba[:, :, 3]
        rgb = rgba[:, :, :3]
        
        pixels_recolored = 0
        
        # Find pixels that are still marked as tendrils
        for y in range(height):
            for x in range(width):
                if alpha[y, x] in [self.HORIZONTAL_THIN, self.VERTICAL_THIN, self.BOTH_THIN]:
                    # Find adjacent non-tendril color
                    new_color = self._find_adjacent_color(rgba, x, y)
                    if new_color is not None:
                        rgb[y, x] = new_color
                        alpha[y, x] = self.NORMAL_PIXEL
                        pixels_recolored += 1
                    else:
                        # Fallback: use a default color
                        rgb[y, x] = [128, 128, 128]  # Gray
                        alpha[y, x] = self.NORMAL_PIXEL
                        pixels_recolored += 1
        
        return pixels_recolored


# Convenience function for easy usage
def trim_tendrils(rgba: np.ndarray, threshold: int, max_iterations: int = 30) -> Tuple[np.ndarray, int, str]:
    """
    Convenience function to trim tendrils from an image.
    
    Args:
        rgba: Input image as RGBA numpy array
        threshold: Maximum thickness for a pixel to be considered a tendril
        max_iterations: Maximum number of iterations to perform
        
    Returns:
        Tuple of (processed_image, iterations_used, status_message)
    """
    trimmer = TendrilTrimmer()
    return trimmer.trim_tendrils(rgba, threshold, max_iterations)