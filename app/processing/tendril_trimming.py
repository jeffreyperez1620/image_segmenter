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
import time
from typing import Tuple, List, Dict, Optional


class TendrilTrimmer:
    """Main class for tendril trimming operations using alpha sentinel values."""
    
    def __init__(self):
        """Initialize the tendril trimmer."""
        # Sentinel values for alpha channel
        self.NORMAL_PIXEL = 10      # Not a tendril or processed tendril
        self.HORIZONTAL_THIN = 11   # Horizontally thin
        self.VERTICAL_THIN = 12     # Vertically thin
        self.BOTH_THIN = 13         # Both horizontally and vertically thin
        
    def log_debug(self,message):
        with open('tendril_debug.log', 'a') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
            f.flush()
    
    def trim_tendrils(self, rgba: np.ndarray, threshold: int, max_iterations: int = 30, progress_callback: Optional[callable] = None) -> Tuple[np.ndarray, int, str]:
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
        
        # Set up transient state for this operation
        self._rgba = result
        self._alpha = result[:, :, 3]
        self._rgb = result[:, :, :3]
        self._height, self._width = result.shape[:2]
        
        try:
            iteration = 0
            pixels_changed = 0
            total_pixels_changed = 0
            
            # Track changed pixels for incremental evaluation
            changed_pixels: set[Tuple[int, int]] = set()  # Set of (y, x) coordinates
            # Track tendril locations for localized mode
            tendril_locations: set[Tuple[int, int]] = set()  # Set of (y, x) coordinates of known tendrils
            total_pixels = np.sum(alpha > 0)
            localized_threshold = max(100, int(total_pixels * 0.01))  # 1% of image or 100 pixels, whichever is larger
            use_localized_mode = False
            
            # Initial progress update
            if progress_callback:
                progress_callback(0, max_iterations, "Starting tendril cleanup...")
            
            while iteration < max_iterations:
                # 1. Mark tendrils and count them
                # Use incremental evaluation after first iteration
                if iteration == 0:
                    # First iteration: evaluate entire image
                    tendril_count = self._mark_tendrils(threshold)
                else:
                    # Subsequent iterations: use incremental or localized mode
                    if use_localized_mode:
                        # Localized mode: only evaluate regions around known tendrils
                        # Combine changed pixels and known tendril locations
                        evaluation_set = changed_pixels | tendril_locations
                        tendril_count = self._mark_tendrils_localized(threshold, evaluation_set)
                    else:
                        # Incremental mode: evaluate changed pixels + neighbors
                        tendril_count = self._mark_tendrils_incremental(threshold, changed_pixels)
                
                # Update tendril locations for next iteration (after marking, before processing)
                if tendril_count > 0:
                    tendril_locations = self._get_tendril_locations()
                else:
                    tendril_locations.clear()
                
                # Progress update for marking phase
                if progress_callback:
                    mode_str = "localized" if use_localized_mode else ("incremental" if iteration > 0 else "full")
                    progress_callback(iteration, max_iterations, f"Marked {tendril_count} tendril pixels ({mode_str} mode)")
                
                # 2. Early exit if no tendrils found
                if tendril_count == 0:
                    if progress_callback:
                        progress_callback(iteration + 1, max_iterations, "No more tendrils found - cleanup complete")
                    break
                
                # 3. Check if we should switch to localized mode
                if not use_localized_mode and tendril_count < localized_threshold:
                    use_localized_mode = True
                    if progress_callback:
                        progress_callback(iteration, max_iterations, f"Switching to localized mode ({tendril_count} tendrils remaining)")
                    
                # 4. Process tendrils (collect changes, then apply)
                pixels_changed, changed_pixels = self._process_tendrils_priority_queue(threshold)
                total_pixels_changed += pixels_changed
                
                # Progress update for processing phase
                if progress_callback:
                    progress_callback(iteration + 1, max_iterations, f"Iteration {iteration + 1}: Changed {pixels_changed} pixels (Total: {total_pixels_changed})")
                
                # Check for convergence: if no pixels changed, we're done
                if pixels_changed == 0:
                    if progress_callback:
                        progress_callback(iteration + 1, max_iterations, "No pixels changed - cleanup complete")
                    break
                
                iteration += 1
            
            # 4. Restore alpha channel to full opacity
            self._restore_alpha_channel(result)
            
            # Final progress update
            if progress_callback:
                progress_callback(max_iterations, max_iterations, f"Cleanup complete: {total_pixels_changed} pixels changed in {iteration} iterations")
            
            if pixels_changed == 0:
                return result, iteration, f"Converged after {iteration} iterations - no pixels changed (Total: {total_pixels_changed} pixels changed)"
            else:
                return result, iteration, f"Completed after {iteration} iterations (Total: {total_pixels_changed} pixels changed)"
        
        finally:
            # Guaranteed cleanup of transient state
            self._cleanup_transient_state()
    
    def _cleanup_transient_state(self) -> None:
        """Clean up transient state after processing."""
        if hasattr(self, '_rgba'):
            del self._rgba
        if hasattr(self, '_alpha'):
            del self._alpha
        if hasattr(self, '_rgb'):
            del self._rgb
        if hasattr(self, '_height'):
            del self._height
        if hasattr(self, '_width'):
            del self._width
    
    def _mark_tendrils(self, threshold: int) -> int:
        """
        Mark tendrils using alpha sentinel values.
        
        Args:
            threshold: Maximum thickness for a pixel to be considered a tendril
            
        Returns:
            Number of tendril pixels found
        """
        # Convert RGB to single integer representation for efficient comparison
        rgb_int = (self._rgb[:, :, 0].astype(np.uint32) << 16) | (self._rgb[:, :, 1].astype(np.uint32) << 8) | self._rgb[:, :, 2].astype(np.uint32)
        
        tendril_count = 0
        
        # First, set all non-transparent pixels to NORMAL_PIXEL (will be reclassified as tendrils if needed)
        non_transparent = self._alpha > 0
        self._alpha[non_transparent] = self.NORMAL_PIXEL
        
        # For each pixel, check if it's a tendril
        for y in range(self._height):
            for x in range(self._width):
                # Skip transparent pixels
                if not non_transparent[y, x]:
                    continue
                
                # Get current color
                current_color = rgb_int[y, x]
                is_horizontal_thin = False
                is_vertical_thin = False
                
                # Check horizontal thickness
                horizontal_thickness = self._calculate_horizontal_thickness(rgb_int, self._alpha, x, y, current_color, self._width)
                if horizontal_thickness <= threshold:
                    is_horizontal_thin = True
                
                # Check vertical thickness
                vertical_thickness = self._calculate_vertical_thickness(rgb_int, self._alpha, x, y, current_color, self._height)
                if vertical_thickness <= threshold:
                    is_vertical_thin = True
                
                # Mark pixel based on tendril type
                if is_horizontal_thin and is_vertical_thin:
                    self._alpha[y, x] = self.BOTH_THIN
                    tendril_count += 1
                elif is_horizontal_thin:
                    self._alpha[y, x] = self.HORIZONTAL_THIN
                    tendril_count += 1
                elif is_vertical_thin:
                    self._alpha[y, x] = self.VERTICAL_THIN
                    tendril_count += 1
        
        return tendril_count
    
    def _mark_tendrils_incremental(self, threshold: int, changed_pixels: set[Tuple[int, int]], buffer: int = 3) -> int:
        """
        Mark tendrils using incremental evaluation - only evaluates changed pixels and their neighbors.
        
        Args:
            threshold: Maximum thickness for a pixel to be considered a tendril
            changed_pixels: Set of (y, x) coordinates that changed in the previous iteration
            buffer: Number of pixels around changed pixels to also evaluate
            
        Returns:
            Number of tendril pixels found
        """
        # Convert RGB to single integer representation for efficient comparison
        rgb_int = (self._rgb[:, :, 0].astype(np.uint32) << 16) | (self._rgb[:, :, 1].astype(np.uint32) << 8) | self._rgb[:, :, 2].astype(np.uint32)
        
        # Build set of pixels to evaluate: changed pixels + their neighbors + buffer
        pixels_to_evaluate: set[Tuple[int, int]] = set()
        non_transparent = self._alpha > 0
        
        for y, x in changed_pixels:
            # Add the changed pixel itself
            pixels_to_evaluate.add((y, x))
            
            # Add neighbors within buffer distance
            for dy in range(-buffer, buffer + 1):
                for dx in range(-buffer, buffer + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self._height and 0 <= nx < self._width and non_transparent[ny, nx]:
                        pixels_to_evaluate.add((ny, nx))
        
        # Reset previously marked tendrils in the evaluation region to NORMAL_PIXEL
        for y, x in pixels_to_evaluate:
            if non_transparent[y, x]:
                self._alpha[y, x] = self.NORMAL_PIXEL
        
        tendril_count = 0
        
        # Evaluate only the pixels in our set
        for y, x in pixels_to_evaluate:
            if not non_transparent[y, x]:
                continue
            
            # Get current color
            current_color = rgb_int[y, x]
            is_horizontal_thin = False
            is_vertical_thin = False
            
            # Check horizontal thickness
            horizontal_thickness = self._calculate_horizontal_thickness(rgb_int, self._alpha, x, y, current_color, self._width)
            if horizontal_thickness <= threshold:
                is_horizontal_thin = True
            
            # Check vertical thickness
            vertical_thickness = self._calculate_vertical_thickness(rgb_int, self._alpha, x, y, current_color, self._height)
            if vertical_thickness <= threshold:
                is_vertical_thin = True
            
            # Mark pixel based on tendril type
            if is_horizontal_thin and is_vertical_thin:
                self._alpha[y, x] = self.BOTH_THIN
                tendril_count += 1
            elif is_horizontal_thin:
                self._alpha[y, x] = self.HORIZONTAL_THIN
                tendril_count += 1
            elif is_vertical_thin:
                self._alpha[y, x] = self.VERTICAL_THIN
                tendril_count += 1
        
        return tendril_count
    
    def _mark_tendrils_localized(self, threshold: int, changed_pixels: set[Tuple[int, int]], buffer: int = 5) -> int:
        """
        Mark tendrils using localized evaluation - only evaluates regions around known tendril locations.
        
        Args:
            threshold: Maximum thickness for a pixel to be considered a tendril
            changed_pixels: Set of (y, x) coordinates that changed in the previous iteration
            buffer: Number of pixels around changed pixels to evaluate (larger for localized mode)
            
        Returns:
            Number of tendril pixels found
        """
        # Convert RGB to single integer representation for efficient comparison
        rgb_int = (self._rgb[:, :, 0].astype(np.uint32) << 16) | (self._rgb[:, :, 1].astype(np.uint32) << 8) | self._rgb[:, :, 2].astype(np.uint32)
        
        # Build set of pixels to evaluate: changed pixels + larger buffer zone
        pixels_to_evaluate: set[Tuple[int, int]] = set()
        non_transparent = self._alpha > 0
        
        for y, x in changed_pixels:
            # Add neighbors within buffer distance (larger buffer for localized mode)
            for dy in range(-buffer, buffer + 1):
                for dx in range(-buffer, buffer + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self._height and 0 <= nx < self._width and non_transparent[ny, nx]:
                        pixels_to_evaluate.add((ny, nx))
        
        # Also check for any remaining marked tendrils from previous iteration
        # (This is a safety check - in normal operation, tendrils should be processed and unmarked)
        for y in range(self._height):
            for x in range(self._width):
                if self._alpha[y, x] in [self.HORIZONTAL_THIN, self.VERTICAL_THIN, self.BOTH_THIN]:
                    # Add this pixel and its neighbors to evaluation set
                    for dy in range(-buffer, buffer + 1):
                        for dx in range(-buffer, buffer + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self._height and 0 <= nx < self._width and non_transparent[ny, nx]:
                                pixels_to_evaluate.add((ny, nx))
        
        # Reset previously marked tendrils in the evaluation region to NORMAL_PIXEL
        for y, x in pixels_to_evaluate:
            if non_transparent[y, x]:
                self._alpha[y, x] = self.NORMAL_PIXEL
        
        tendril_count = 0
        
        # Evaluate only the pixels in our set
        for y, x in pixels_to_evaluate:
            if not non_transparent[y, x]:
                continue
            
            # Get current color
            current_color = rgb_int[y, x]
            is_horizontal_thin = False
            is_vertical_thin = False
            
            # Check horizontal thickness
            horizontal_thickness = self._calculate_horizontal_thickness(rgb_int, self._alpha, x, y, current_color, self._width)
            if horizontal_thickness <= threshold:
                is_horizontal_thin = True
            
            # Check vertical thickness
            vertical_thickness = self._calculate_vertical_thickness(rgb_int, self._alpha, x, y, current_color, self._height)
            if vertical_thickness <= threshold:
                is_vertical_thin = True
            
            # Mark pixel based on tendril type
            if is_horizontal_thin and is_vertical_thin:
                self._alpha[y, x] = self.BOTH_THIN
                tendril_count += 1
            elif is_horizontal_thin:
                self._alpha[y, x] = self.HORIZONTAL_THIN
                tendril_count += 1
            elif is_vertical_thin:
                self._alpha[y, x] = self.VERTICAL_THIN
                tendril_count += 1
        
        return tendril_count
    
    def _calculate_horizontal_thickness(self, rgb_int: np.ndarray, alpha: np.ndarray, x: int, y: int, current_color: int, width: int) -> int:
        """Calculate horizontal thickness of a pixel."""
        left_dist = 0
        right_dist = 0
        left_hits_transparent = False
        right_hits_transparent = False
        
        # Count distance to left
        for dx in range(1, x + 2):
            if x - dx < 0 or alpha[y, x - dx] == 0:
                left_hits_transparent = True
                break
            if rgb_int[y, x - dx] != current_color:
                break
            left_dist += 1
        
        # Count distance to right
        for dx in range(1, width - x + 1):
            if x + dx >= width or alpha[y, x + dx] == 0:
                right_hits_transparent = True
                break
            if rgb_int[y, x + dx] != current_color:
                break
            right_dist += 1
        
        # If both ends hit transparent pixels (or boundaries), return infinity
        if left_hits_transparent and right_hits_transparent:
            return float('inf')  # Sentinel value: thin only due to transparent borders
        
        # Total thickness is the sum of the distances to the left and right plus 1 for the current pixel
        return left_dist + right_dist + 1
    
    def _calculate_vertical_thickness(self, rgb_int: np.ndarray, alpha: np.ndarray, x: int, y: int, current_color: int, height: int) -> int:
        """Calculate vertical thickness of a pixel."""
        up_dist = 0
        down_dist = 0
        up_hits_transparent = False
        down_hits_transparent = False
        
        # Count distance up
        for dy in range(1, y + 2):
            if y - dy < 0 or alpha[y - dy, x] == 0:
                up_hits_transparent = True
                break
            if rgb_int[y - dy, x] != current_color:
                break
            up_dist += 1
        
        # Count distance down
        for dy in range(1, height - y + 1):
            if y + dy >= height or alpha[y + dy, x] == 0:
                down_hits_transparent = True
                break
            if rgb_int[y + dy, x] != current_color:
                break
            down_dist += 1
        
        # If both ends hit transparent pixels (or boundaries), return infinity
        if up_hits_transparent and down_hits_transparent:
            return float('inf')  # Sentinel value: thin only due to transparent borders
        
        # Total thickness is the sum of the distances to the up and down plus 1 for the current pixel
        return up_dist + down_dist + 1
    
    def _process_tendrils(self, threshold: int) -> None:
        """
        Process all tendrils by collecting changes and applying them together.
        
        Args:
            threshold: Maximum thickness for a pixel to be considered a tendril
        """
        # Process horizontal tendrils first (HORIZONTAL_THIN and BOTH_THIN pixels)
        self._process_horizontal_tendrils(threshold)
        
        # Then process vertical tendrils (VERTICAL_THIN and BOTH_THIN pixels)
        self._process_vertical_tendrils(threshold)
    
    def _process_tendrils_priority_queue(self, threshold: int) -> Tuple[int, set[Tuple[int, int]]]:
        """
        Process all tendrils using a priority queue approach.
        
        This is a new implementation that processes tendrils iteratively,
        starting with the most obvious cases (pixels surrounded by many
        neighbors of the same color) and re-evaluating neighbors as
        colors change.
        
        Args:
            threshold: Maximum thickness for a pixel to be considered a tendril
            
        Returns:
            Tuple of (number of pixels changed, set of changed pixel coordinates (y, x))
        """
        import heapq
        from collections import defaultdict
        
        # Create priority queue (max-heap using negative priorities)
        # Each item is (-priority, y, x, majority_color)
        priority_queue = []
        
        # Track the latest priority for each pixel to avoid processing outdated entries
        pixel_priorities = {}
        
        # Track number of pixels that actually change color
        pixels_changed = 0
        changed_pixels: set[Tuple[int, int]] = set()
        
        # Find all tendril pixels and calculate their priorities
        for y in range(self._height):
            for x in range(self._width):
                if self._alpha[y, x] in [self.HORIZONTAL_THIN, self.VERTICAL_THIN, self.BOTH_THIN]:
                    priority, majority_color = self._find_majority_adjacent_color(x, y)
                    if priority > 0:  # Only add if there are eligible neighbors
                        heapq.heappush(priority_queue, (-priority, y, x, tuple(majority_color)))
                        pixel_priorities[(y, x)] = priority
        
        # Process the priority queue
        while priority_queue:
            # Get the highest priority pixel
            neg_priority, y, x, majority_color = heapq.heappop(priority_queue)
            priority = -neg_priority
            
            # Skip if this pixel is no longer a tendril (may have been processed already)
            if self._alpha[y, x] not in [self.HORIZONTAL_THIN, self.VERTICAL_THIN, self.BOTH_THIN]:
                continue
            
            # Skip if this is an outdated entry (pixel has been re-evaluated with higher priority)
            if (y, x) in pixel_priorities and pixel_priorities[(y, x)] > priority:
                continue
            
            # Check if color actually changes
            current_color = self._rgb[y, x].copy()
            new_color = np.array(majority_color, dtype=np.uint8)
            
            if not np.array_equal(current_color, new_color):
                self._rgb[y, x] = new_color
                pixels_changed += 1
                changed_pixels.add((y, x))
            
            # Always de-mark the pixel (it's been processed)
            self._alpha[y, x] = self.NORMAL_PIXEL
            
            # Re-evaluate neighbors that are also tendrils
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:  # Skip the current pixel
                        continue
                    
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if not (0 <= ny < self._height and 0 <= nx < self._width):
                        continue
                    
                    # Check if neighbor is a tendril
                    if self._alpha[ny, nx] in [self.HORIZONTAL_THIN, self.VERTICAL_THIN, self.BOTH_THIN]:
                        # Recalculate priority for this neighbor
                        new_priority, new_majority_color = self._find_majority_adjacent_color(nx, ny)
                        
                        # Add to queue with new priority if it has eligible neighbors
                        if new_priority > 0:
                            heapq.heappush(priority_queue, (-new_priority, ny, nx, tuple(new_majority_color)))
                            pixel_priorities[(ny, nx)] = new_priority
        
        return pixels_changed, changed_pixels
    
    def _find_majority_adjacent_color(self, x: int, y: int) -> tuple[int, np.ndarray]:
        """
        Find the majority color of non-tendril neighbors that are different from the current pixel.
        
        Args:
            x, y: Pixel coordinates
            
        Returns:
            Tuple of (count, majority_color) where count is the number of neighbors of the majority color.
            If no eligible neighbors found, returns (0, current_color).
        """
        current_color = self._rgb[y, x]
        
        # Check 4-neighbors (cardinal directions only)
        neighbors = [
            (y-1, x),   # up
            (y, x+1),   # right
            (y+1, x),   # down
            (y, x-1)    # left
        ]
        
        # Count colors of eligible neighbors
        neighbor_color_counts = {}
        
        for ny, nx in neighbors:
            # Check bounds
            if not (0 <= ny < self._height and 0 <= nx < self._width):
                continue
                
            neighbor_alpha = self._alpha[ny, nx]
            neighbor_color = self._rgb[ny, nx]
            
            # Skip if it's not a normal pixel (i.e., it's a tendril or transparent)
            if neighbor_alpha != self.NORMAL_PIXEL:
                continue
                
            # Skip if same color as current pixel
            # if np.array_equal(neighbor_color, current_color):
            #     continue
            
            # Count this valid neighbor color
            color_key = tuple(neighbor_color)
            neighbor_color_counts[color_key] = neighbor_color_counts.get(color_key, 0) + 1
            
        # If no valid neighbors found, return 0 priority
        if not neighbor_color_counts:
            return 0, current_color
        
        # Find the majority color and its count
        majority_color_key = max(neighbor_color_counts, key=neighbor_color_counts.get)
        majority_count = neighbor_color_counts[majority_color_key]
        majority_color = np.array(majority_color_key, dtype=np.uint8)

        return majority_count, majority_color
    
    def _process_horizontal_tendrils(self, threshold: int) -> None:
        """Process horizontal tendrils by scanning horizontally. Handles HORIZONTAL_THIN and BOTH_THIN pixels."""
        
        # Collect all changes before applying them
        changes = {}  # (y, x) -> new_color
        
        for y in range(self._height):
            x = 0
            while x < self._width:
                if self._alpha[y, x] in [self.HORIZONTAL_THIN, self.BOTH_THIN]:
                    # Found start of horizontal tendril
                    scan_line_info = self._scan_horizontal_line(x, y)
                    if scan_line_info:
                        # Scan line info is a tuple of (start_x, end_x, majority_color)
                        start_x, end_x, majority_color = scan_line_info
                        line_length = end_x - start_x + 1
                        
                        #if line_length > threshold:
                        #    # This indicates that multiple different colored tendrils are connected
                        #    # Scan line is long enough - set all pixels to majority color
                        #    for px in range(start_x, end_x + 1):
                        #        changes[(y, px)] = majority_color
                        #else:
                            # Scan line is too short - use adjacent non-tendril colors
                        for px in range(start_x, end_x + 1):
                            count, new_color = self._find_majority_adjacent_color(px, y)
                            if count > 0:
                                changes[(y, px)] = new_color
                            else:
                                # no adjacent non-tendril color found, use the current color
                                # re-evaluate if this pixel is a tendril on next iteration
                                changes[(y, px)] = self._rgb[y, px]
                    
                        x = end_x + 1  # Skip to end of scan line
                    else:
                        # This should not happen - scan_line_info should always be returned for a tendril pixel
                        print(f"WARNING: _scan_horizontal_line returned None for tendril pixel at ({y}, {x})")
                        x += 1
                else:
                    # Not a tendril pixel - move to next pixel
                    x += 1
        
        # Apply all changes
        for (y, x), new_color in changes.items():
            self._rgb[y, x] = new_color
            self._alpha[y, x] = self.NORMAL_PIXEL  # Mark as processed
    
    def _process_vertical_tendrils(self, threshold: int) -> None:
        """Process vertical tendrils by scanning vertically. Handles VERTICAL_THIN and BOTH_THIN pixels."""
        
        # Collect all changes before applying them
        changes = {}  # (y, x) -> new_color
        
        for x in range(self._width):
            y = 0
            while y < self._height:
                if self._alpha[y, x] in [self.VERTICAL_THIN, self.BOTH_THIN]:
                    # Found start of vertical tendril
                    scan_line_info = self._scan_vertical_line(x, y)
                    if scan_line_info:
                        start_y, end_y, majority_color = scan_line_info
                        
                        #if end_y - start_y + 1 > threshold:
                        #    # Scan line is long enough - set all pixels to majority color
                        #    for py in range(start_y, end_y + 1):
                        #        changes[(py, x)] = majority_color
                        #else:
                            # Scan line is too short - use adjacent non-tendril colors
                        for py in range(start_y, end_y + 1):
                            count, new_color = self._find_majority_adjacent_color(x, py)
                            if count > 0:
                                changes[(py, x)] = new_color 
                            else:
                                # no adjacent non-tendril color found, use the current color
                                # re-evaluate if this pixel is a tendril on next iteration
                                changes[(py, x)] = self._rgb[py, x]
                    
                        y = end_y + 1  # Skip to end of scan line
                    else:
                        # This should not happen - scan_line_info should always be returned for a tendril pixel
                        print(f"WARNING: _scan_vertical_line returned None for tendril pixel at ({y}, {x})")
                        y += 1
                else:
                    y += 1
        
        # Apply all changes
        for (y, x), new_color in changes.items():
            self._rgb[y, x] = new_color
            self._alpha[y, x] = self.NORMAL_PIXEL  # Mark as processed
    
    def _scan_horizontal_line(self, start_x: int, y: int) -> Optional[Tuple[int, int, np.ndarray]]:
        """Scan horizontally to find the extent of a tendril line and determine majority color."""
        # Example: [NORMAL_PIXEL], red, red, blue, red, [NORMAL_PIXEL]
        # all 4 pixels are tendril pixels, but the majority color is red
        # if there is a tie, the leftmost color is used
        # if there is no tendril pixel in the scan line, return None
        # if there is only one tendril pixel in the scan line, return the color of that pixel
        # if there are multiple tendril pixels in the scan line, return the color that appears most frequently
        
        # Find the extent of the tendril line
        end_x = start_x
        while end_x + 1 < self._width and self._alpha[y, end_x + 1] in [self.HORIZONTAL_THIN, self.BOTH_THIN]:
            end_x += 1
        
        # Count colors in the scan line
        color_counts = {}
        for x in range(start_x, end_x + 1):
            color_key = tuple(self._rgb[y, x])
            color_counts[color_key] = color_counts.get(color_key, 0) + 1
        
        if not color_counts:
            return None
        
        # Find majority color
        majority_color = self._find_majority_color(color_counts)
        
        return start_x, end_x, majority_color
    
    def _scan_vertical_line(self, x: int, start_y: int) -> Optional[Tuple[int, int, np.ndarray]]:
        """Scan vertically to find the extent of a tendril line and determine majority color."""
        # Example: [NORMAL_PIXEL], red, red, blue, red, [NORMAL_PIXEL]
        # all 4 pixels are tendril pixels, but the majority color is red
        # if there is a tie, the leftmost color is used
        # if there is no tendril pixel in the scan line, return None
        # if there is only one tendril pixel in the scan line, return the color of that pixel
        # if there are multiple tendril pixels in the scan line, return the color that appears most frequently
        
        # Find the extent of the tendril line
        end_y = start_y
        while end_y + 1 < self._height and self._alpha[end_y + 1, x] in [self.VERTICAL_THIN, self.BOTH_THIN]:
            end_y += 1
        
        # Count colors in the scan line
        color_counts = {}
        for y in range(start_y, end_y + 1):
            color_key = tuple(self._rgb[y, x])
            color_counts[color_key] = color_counts.get(color_key, 0) + 1
        
        if not color_counts:
            return None
        
        # Find majority color
        majority_color = self._find_majority_color(color_counts)
        
        return start_y, end_y, majority_color
    
    def _find_majority_color(self, color_counts: Dict[Tuple[int, ...], int]) -> np.ndarray:
        """Find the majority color from a dictionary of color counts with tie-breaking."""
        if not color_counts:
            raise ValueError("color_counts cannot be empty")
        
        # Find the majority color (use first encountered in case of tie)
        majority_color = max(color_counts, key=lambda k: (color_counts[k], -list(color_counts.keys()).index(k)))
        
        return np.array(majority_color, dtype=np.uint8)
    
    def _get_tendril_locations(self) -> set[Tuple[int, int]]:
        """
        Get the set of coordinates of all currently marked tendril pixels.
        
        Returns:
            Set of (y, x) coordinates of tendril pixels
        """
        tendril_locations: set[Tuple[int, int]] = set()
        for y in range(self._height):
            for x in range(self._width):
                if self._alpha[y, x] in [self.HORIZONTAL_THIN, self.VERTICAL_THIN, self.BOTH_THIN]:
                    tendril_locations.add((y, x))
        return tendril_locations
    
    def _restore_alpha_channel(self, result: np.ndarray) -> None:
        """Restore alpha channel to full opacity for all originally non-transparent pixels."""
        # Set alpha to 255 for all pixels that were originally non-transparent
        result_non_transparent = result[:, :, 3] > 0
        result[result_non_transparent, 3] = 255
    

    def _mark_tendrils_for_iteration(self, rgba: np.ndarray, threshold: int) -> int:
        """
        Mark tendrils for one iteration.
        """
        # Set up transient state for this operation
        self._rgba = rgba
        self._alpha = rgba[:, :, 3]
        self._rgb = rgba[:, :, :3]
        self._height, self._width = rgba.shape[:2]

        # Mark tendrils
        tendril_count = self._mark_tendrils(threshold)
        
        return tendril_count
    
    # Backward compatibility methods for the test app
    def _trim_tendrils_in_iteration(self, rgba: np.ndarray, threshold: int) -> int:
        """
        Backward compatibility method for the test app.
        This method runs one iteration of the new algorithm and returns the number of pixels changed.
        """
        # Set up transient state for this operation
        self._rgba = rgba
        self._alpha = rgba[:, :, 3]
        self._rgb = rgba[:, :, :3]
        self._height, self._width = rgba.shape[:2]
        
        # Mark tendrils
        tendril_count = self._mark_tendrils(threshold)
        
        if tendril_count == 0:
            return 0
        
        # Process tendrils
        pixels_changed, _ = self._process_tendrils_priority_queue(threshold)
        
        return pixels_changed


# Convenience function for easy usage
def trim_tendrils(rgba: np.ndarray, threshold: int, max_iterations: int = 30, progress_callback: Optional[callable] = None) -> Tuple[np.ndarray, int, str]:
    """
    Convenience function to trim tendrils from an image.
    
    Args:
        rgba: Input image as RGBA numpy array
        threshold: Maximum thickness for a pixel to be considered a tendril
        max_iterations: Maximum number of iterations to perform
        progress_callback: Optional callback function for progress updates (current, total, message)
        
    Returns:
        Tuple of (processed_image, iterations_used, status_message)
    """
    trimmer = TendrilTrimmer()
    return trimmer.trim_tendrils(rgba, threshold, max_iterations, progress_callback)