from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2 as cv
from collections import defaultdict


def analyze_regions(rgba: np.ndarray, min_size_threshold: int = 100, connectivity: int = 8) -> Dict:
    """
    Analyze regions in a color-simplified RGBA image.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image, shape (H, W, 4)
    min_size_threshold: int
        Minimum size threshold for considering regions as "small"
        
    Returns
    -------
    Dict
        Region analysis statistics
    """
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba must be HxWx4 uint8")
    
    # Get RGB channels and create a unique color mapping
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    
    # Only analyze non-transparent pixels
    non_transparent = alpha > 0
    if not np.any(non_transparent):
        return {
            'total_regions': 0,
            'small_regions': 0,
            'largest_region_size': 0,
            'smallest_region_size': 0,
            'size_distribution': {},
            'region_colors': [],
            'region_sizes': [],
            'all_regions': []
        }
    
    # Create a mask for non-transparent pixels
    mask = non_transparent.astype(np.uint8) * 255
    
    # Find connected components for each unique color
    unique_colors = np.unique(rgb[non_transparent].reshape(-1, 3), axis=0)
    
    all_regions = []
    region_colors = []
    region_sizes = []
    small_regions_count = 0
    
    for color in unique_colors:
        # Create binary mask for this color
        color_mask = np.all(rgb == color, axis=2) & non_transparent
        color_mask = color_mask.astype(np.uint8) * 255
        
        # Find connected components for this color
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(color_mask, connectivity=connectivity)
        
        # Process each component (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            if area > 0:  # Only include non-empty regions
                region_info = {
                    'color': tuple(color),
                    'size': int(area),
                    'label': i,
                    'color_mask': color_mask,
                    'labels': labels,
                    'component_id': i,
                    'bbox': (
                        stats[i, cv.CC_STAT_LEFT],
                        stats[i, cv.CC_STAT_TOP], 
                        stats[i, cv.CC_STAT_WIDTH],
                        stats[i, cv.CC_STAT_HEIGHT]
                    )
                }
                all_regions.append(region_info)
                region_colors.append(tuple(color))
                region_sizes.append(int(area))
                
                if area < min_size_threshold:
                    small_regions_count += 1
    
    # Calculate statistics
    if not region_sizes:
        return {
            'total_regions': 0,
            'small_regions': 0,
            'largest_region_size': 0,
            'smallest_region_size': 0,
            'size_distribution': {},
            'region_colors': [],
            'region_sizes': [],
            'all_regions': []
        }
    
    total_regions = len(region_sizes)
    largest_region = max(region_sizes)
    smallest_region = min(region_sizes)
    
    # Size distribution
    size_distribution = defaultdict(int)
    for size in region_sizes:
        if size < 50:
            size_distribution['< 50'] += 1
        elif size < 100:
            size_distribution['50-99'] += 1
        elif size < 200:
            size_distribution['100-199'] += 1
        elif size < 500:
            size_distribution['200-499'] += 1
        else:
            size_distribution['500+'] += 1
    
    return {
        'total_regions': total_regions,
        'small_regions': small_regions_count,
        'largest_region_size': largest_region,
        'smallest_region_size': smallest_region,
        'size_distribution': dict(size_distribution),
        'region_colors': region_colors,
        'region_sizes': region_sizes,
        'all_regions': all_regions
    }


def find_neighboring_colors_for_component(rgba: np.ndarray, component_mask: np.ndarray, connectivity: int = 8) -> List[Tuple[int, int, int]]:
    """
    Find colors that are adjacent to a specific connected component.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image
    component_mask: np.ndarray
        Boolean mask for the specific component
        
    Returns
    -------
    List[Tuple[int, int, int]]
        List of neighboring colors
    """
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    
    # Dilate the component mask to find adjacent pixels
    # Use connectivity-appropriate kernel
    if connectivity == 4:
        # 4-way connectivity: only horizontal and vertical neighbors
        kernel = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)
    else:
        # 8-way connectivity: all neighbors including diagonal
        kernel = np.ones((3, 3), np.uint8)
    
    dilated_mask = cv.dilate(component_mask.astype(np.uint8), kernel, iterations=1)
    
    # Find pixels that are adjacent but not part of the component
    adjacent_mask = (dilated_mask > 0) & ~component_mask & (alpha > 0)
    
    # Get unique colors in adjacent areas
    if np.any(adjacent_mask):
        adjacent_colors = np.unique(rgb[adjacent_mask].reshape(-1, 3), axis=0)
        return [tuple(color) for color in adjacent_colors]
    else:
        return []


def find_neighboring_colors(rgba: np.ndarray, target_color: Tuple[int, int, int], connectivity: int = 8) -> List[Tuple[int, int, int]]:
    """
    Find colors that are adjacent to the target color in the image.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image
    target_color: Tuple[int, int, int]
        RGB color to find neighbors for
        
    Returns
    -------
    List[Tuple[int, int, int]]
        List of neighboring colors
    """
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    
    # Create mask for target color
    target_mask = np.all(rgb == target_color, axis=2) & (alpha > 0)
    
    # Dilate the mask to find adjacent pixels
    # Use connectivity-appropriate kernel
    if connectivity == 4:
        # 4-way connectivity: only horizontal and vertical neighbors
        kernel = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)
    else:
        # 8-way connectivity: all neighbors including diagonal
        kernel = np.ones((3, 3), np.uint8)
    
    dilated_mask = cv.dilate(target_mask.astype(np.uint8), kernel, iterations=1)
    
    # Find pixels that are adjacent but not part of the target color
    adjacent_mask = (dilated_mask > 0) & ~target_mask & (alpha > 0)
    
    # Get unique colors in adjacent areas
    adjacent_colors = np.unique(rgb[adjacent_mask].reshape(-1, 3), axis=0)
    
    return [tuple(color) for color in adjacent_colors]


def calculate_merge_score(
    small_region: dict, 
    neighbor_color: Tuple[int, int, int], 
    neighbor_region_size: int,
    image_context: dict,
    weights: dict = None
) -> float:
    """
    Calculate a score for merging a small region into a neighbor color.
    
    Parameters
    ----------
    small_region: dict
        Information about the small region
    neighbor_color: Tuple[int, int, int]
        RGB color of the potential merge target
    neighbor_region_size: int
        Size of the neighbor region in pixels
    image_context: dict
        Context information about the image
    weights: dict, optional
        Weights for different factors
        
    Returns
    -------
    float
        Score between 0 and 1, higher is better
    """
    if weights is None:
        weights = {
            'color': 0.4,      # Color similarity
            'spatial': 0.3,    # Spatial proximity
            'frequency': 0.2,  # Color frequency
            'size': 0.1        # Region size
        }
    
    # Factor 1: Color similarity (0-1, higher is better)
    color_sim = 1.0 - color_distance(small_region['color'], neighbor_color)
    
    # Factor 2: Spatial proximity (0-1, higher is closer)
    # For now, use a simple approximation - could be improved with actual centroids
    spatial_prox = 0.5  # Placeholder - would need centroid calculation
    
    # Factor 3: Frequency (0-1, higher is more common)
    total_pixels = image_context.get('total_pixels', 1)
    neighbor_pixel_count = image_context.get('color_counts', {}).get(neighbor_color, 1)
    frequency = min(1.0, neighbor_pixel_count / (total_pixels * 0.1))  # Normalize
    
    # Factor 4: Region size (prefer larger neighbors)
    max_region_size = image_context.get('max_region_size', 1)
    size_factor = min(1.0, neighbor_region_size / max_region_size)
    
    # Weighted combination
    score = (weights['color'] * color_sim + 
             weights['spatial'] * spatial_prox + 
             weights['frequency'] * frequency + 
             weights['size'] * size_factor)
    
    return score


def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Calculate perceptual color distance between two RGB colors.
    Returns a value between 0 and 1, where 0 is identical and 1 is maximally different.
    """
    # Convert to LAB color space for better perceptual distance
    import cv2 as cv
    
    # Create small images for color conversion
    img1 = np.array([[color1]], dtype=np.uint8)
    img2 = np.array([[color2]], dtype=np.uint8)
    
    # Convert RGB to LAB
    lab1 = cv.cvtColor(img1, cv.COLOR_RGB2LAB)
    lab2 = cv.cvtColor(img2, cv.COLOR_RGB2LAB)
    
    # Calculate Euclidean distance in LAB space using float64 to avoid overflow
    l1, a1, b1 = lab1[0, 0].astype(np.float64)
    l2, a2, b2 = lab2[0, 0].astype(np.float64)
    
    # Calculate distance with proper bounds
    l_diff = l1 - l2
    a_diff = a1 - a2
    b_diff = b1 - b2
    
    # Normalize to 0-1 range (LAB values are roughly 0-255)
    # Use a more conservative normalization to avoid overflow
    max_lab_distance = 255.0 * np.sqrt(3)  # Maximum possible distance in LAB space
    distance = np.sqrt(l_diff**2 + a_diff**2 + b_diff**2) / max_lab_distance
    
    return min(1.0, max(0.0, distance))


def merge_small_regions(
    rgba: np.ndarray, 
    min_size: int, 
    merge_callback: Optional[callable] = None,
    auto_merge_threshold: float = 0.7,
    merge_weights: dict = None,
    progress_callback: Optional[callable] = None,
    connectivity: int = 8
) -> np.ndarray:
    """
    Merge small regions into larger neighboring regions using hybrid scoring.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image
    min_size: int
        Minimum region size threshold
    merge_callback: callable, optional
        Callback function for user to choose merge color
    auto_merge_threshold: float
        Confidence threshold for automatic merging (0-1)
    merge_weights: dict, optional
        Weights for different merge factors
    progress_callback: callable, optional
        Callback function for progress updates (current, total, message)
        
    Returns
    -------
    np.ndarray
        Image with small regions merged
    """
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Analyze regions with the minimum size threshold
    if progress_callback:
        progress_callback(0, 100, "Analyzing regions...")
    
    stats = analyze_regions(rgba, min_size, connectivity)
    all_regions = stats.get('all_regions', [])
    
    if progress_callback:
        progress_callback(10, 100, "Building image context...")
    
    # Initial image context (will be updated each pass)
    image_context = {
        'total_pixels': np.sum(alpha > 0),
        'max_region_size': 1,
        'color_counts': {}
    }
    
    # Perform multiple passes to ensure all small regions are handled
    max_passes = 20  # Increased to allow complete cleanup
    pass_num = 0
    total_auto_merged = 0
    total_user_decisions = 0
    previous_small_count = float('inf')  # Track progress to detect when no more progress is possible
    
    while pass_num < max_passes:
        pass_num += 1
        
        # Re-analyze regions after each pass
        if progress_callback:
            progress_callback(20 + (pass_num - 1) * 25, 100, f"Pass {pass_num}: Analyzing regions...")
        
        stats = analyze_regions(result, min_size, connectivity)
        all_regions = stats.get('all_regions', [])
        
        # Update image context for this pass
        image_context['max_region_size'] = max([r['size'] for r in all_regions]) if all_regions else 1
        image_context['color_counts'] = {}
        
        # Count pixels for each color
        for region in all_regions:
            color = region['color']
            if color not in image_context['color_counts']:
                image_context['color_counts'][color] = 0
            image_context['color_counts'][color] += region['size']
        
        # Find small regions (individual connected components)
        small_regions = [region for region in all_regions if region['size'] < min_size]
        
        if not small_regions:
            # No more small regions found
            break
        
        # Check if we're making progress
        current_small_count = len(small_regions)
        if current_small_count >= previous_small_count:
            # No progress made - stop to prevent infinite loop
            print(f"No progress made in pass {pass_num}, stopping early")
            break
        previous_small_count = current_small_count
            
        if progress_callback:
            progress_callback(20 + (pass_num - 1) * 25, 100, f"Pass {pass_num}: Found {len(small_regions)} small regions to process...")
        
        # Debug: Print region information (only for first few passes)
        if pass_num <= 3:
            print(f"Pass {pass_num}: Total regions: {len(all_regions)}, Small regions: {len(small_regions)}")
            if all_regions:
                sizes = [r['size'] for r in all_regions]
                print(f"Region sizes: min={min(sizes)}, max={max(sizes)}, threshold={min_size}")
        
        auto_merged = 0
        user_decisions = 0
        total_regions = len(small_regions)
        
        for i, region in enumerate(small_regions):
            # Update progress
            if progress_callback:
                progress = 20 + (pass_num - 1) * 25 + int((i / total_regions) * 20)  # 20-90% for processing
                progress_callback(progress, 100, f"Pass {pass_num}: Processing region {i+1}/{total_regions}...")
            
            target_color = region['color']
            region_size = region['size']
            labels = region['labels']
            component_id = region['component_id']
        
            # Create mask for this specific connected component
            component_mask = (labels == component_id) & (alpha > 0)
            
            if not np.any(component_mask):
                continue
            
            # Find neighboring colors by looking at pixels adjacent to this component
            neighbor_colors = find_neighboring_colors_for_component(result, component_mask, connectivity)
            
            if not neighbor_colors:
                # If no neighbors found, try to find the most common color in the image
                if image_context['color_counts']:
                    most_common_color = max(image_context['color_counts'].items(), key=lambda x: x[1])[0]
                    neighbor_colors = [most_common_color]
                else:
                    continue
            
            # Calculate scores for each neighbor
            neighbor_scores = []
            for neighbor_color in neighbor_colors:
                # Get size of this neighbor region
                neighbor_mask = np.all(rgb == neighbor_color, axis=2) & (alpha > 0)
                neighbor_size = np.sum(neighbor_mask)
                
                score = calculate_merge_score(region, neighbor_color, neighbor_size, image_context, merge_weights)
                neighbor_scores.append((neighbor_color, score))
            
            # Sort by score (highest first)
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            best_color, best_score = neighbor_scores[0]
            
            # Decide whether to auto-merge or ask user
            should_auto_merge = (len(neighbor_colors) == 1 or 
                               (len(neighbor_colors) > 1 and best_score >= auto_merge_threshold))
            
            if should_auto_merge:
                # Auto-merge
                merge_rgb = best_color
                auto_merged += 1
            else:
                # Ask user for decision
                if merge_callback:
                    from PySide6.QtGui import QColor
                    target_qcolor = QColor(target_color[0], target_color[1], target_color[2])
                    neighbor_qcolors = [QColor(c[0], c[1], c[2]) for c in neighbor_colors]
                    
                    # Get bounding box for this specific component
                    bbox = get_component_bounding_box(component_mask, buffer=10)
                    
                    merge_color = merge_callback(target_qcolor, neighbor_qcolors, result, bbox)
                    if merge_color is None:
                        # User cancelled - return None to indicate entire operation should be cancelled
                        return None
                    
                    merge_rgb = (merge_color.red(), merge_color.green(), merge_color.blue())
                    user_decisions += 1
                else:
                    # Fallback to best score
                    merge_rgb = best_color
                    auto_merged += 1
            
            # Apply the merge to this specific component
            rgb[component_mask] = merge_rgb
        
        # Update totals for this pass
        total_auto_merged += auto_merged
        total_user_decisions += user_decisions
        
        if progress_callback:
            progress_callback(20 + pass_num * 25, 100, f"Pass {pass_num} complete: {auto_merged} auto-merged, {user_decisions} user decisions")
    
    if progress_callback:
        progress_callback(100, 100, f"Complete: {total_auto_merged} auto-merged, {total_user_decisions} user decisions in {pass_num} passes")
    
    print(f"Region cleanup complete: {total_auto_merged} auto-merged, {total_user_decisions} user decisions in {pass_num} passes")
    
    # Ensure the result is contiguous
    return np.ascontiguousarray(result)


def flood_fill_region(
    rgba: np.ndarray, 
    seed_point: Tuple[int, int], 
    fill_color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Flood fill a region with a specific color.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image
    seed_point: Tuple[int, int]
        Starting point for flood fill (x, y)
    fill_color: Tuple[int, int, int]
        RGB color to fill with
        
    Returns
    -------
    np.ndarray
        Image with region flood filled
    """
    # Ensure the input is contiguous and has the right dtype
    result = np.ascontiguousarray(rgba.copy(), dtype=np.uint8)
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    x, y = seed_point
    
    # Check bounds
    if x < 0 or y < 0 or x >= rgb.shape[1] or y >= rgb.shape[0]:
        return result
    
    # Only flood fill non-transparent pixels
    if alpha[y, x] == 0:
        return result
    
    # Get the original color at the seed point
    original_color = tuple(rgb[y, x])
    
    # Create mask for flood fill - ensure it's contiguous
    mask = np.zeros((rgb.shape[0] + 2, rgb.shape[1] + 2), dtype=np.uint8)
    
    # Perform flood fill with proper array handling
    try:
        cv.floodFill(
            rgb, 
            mask, 
            (x, y), 
            fill_color,
            loDiff=(0, 0, 0),
            upDiff=(0, 0, 0),
            flags=cv.FLOODFILL_FIXED_RANGE
        )
    except cv.error as e:
        # If OpenCV floodFill fails, try a manual flood fill implementation
        print(f"OpenCV floodFill failed: {e}, using manual implementation")
        result = _manual_flood_fill(result, seed_point, fill_color)
    
    return result


def _manual_flood_fill(
    rgba: np.ndarray, 
    seed_point: Tuple[int, int], 
    fill_color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Manual flood fill implementation as fallback.
    """
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    x, y = seed_point
    
    # Check bounds
    if x < 0 or y < 0 or x >= rgb.shape[1] or y >= rgb.shape[0]:
        return result
    
    # Only flood fill non-transparent pixels
    if alpha[y, x] == 0:
        return result
    
    # Get the original color at the seed point
    original_color = tuple(rgb[y, x])
    
    # Simple flood fill using a stack
    stack = [(x, y)]
    visited = set()
    
    while stack:
        cx, cy = stack.pop()
        
        if (cx, cy) in visited:
            continue
            
        if (cx < 0 or cx >= rgb.shape[1] or 
            cy < 0 or cy >= rgb.shape[0] or 
            alpha[cy, cx] == 0):
            continue
            
        if tuple(rgb[cy, cx]) != original_color:
            continue
            
        visited.add((cx, cy))
        rgb[cy, cx] = fill_color
        
        # Add neighbors to stack
        stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    
    return result


def get_component_bounding_box(component_mask: np.ndarray, buffer: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    Get the bounding box of a specific connected component with a buffer around it.
    
    Parameters
    ----------
    component_mask: np.ndarray
        Boolean mask for the specific component
    buffer: int
        Buffer size in pixels around the region
        
    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        Bounding box as (x, y, width, height) or None if region not found
    """
    if not np.any(component_mask):
        return None
    
    # Find bounding box
    rows = np.any(component_mask, axis=1)
    cols = np.any(component_mask, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add buffer
    h, w = component_mask.shape
    x_min = max(0, x_min - buffer)
    y_min = max(0, y_min - buffer)
    x_max = min(w, x_max + buffer + 1)
    y_max = min(h, y_max + buffer + 1)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def get_region_boundaries(rgba: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Get region boundaries for visualization.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image
        
    Returns
    -------
    np.ndarray
        RGBA image showing region boundaries (white boundaries on transparent background)
    """
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    
    # Only process non-transparent pixels
    non_transparent = alpha > 0
    if not np.any(non_transparent):
        return np.zeros((rgba.shape[0], rgba.shape[1], 4), dtype=np.uint8)
    
    # Create a mask for non-transparent pixels
    mask = non_transparent.astype(np.uint8) * 255
    
    # Method: Use morphological operations to find boundaries between different colors
    # This works better for cleaned images where regions are more uniform
    
    # Convert to grayscale for processing
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    
    # Apply morphological gradient to find boundaries
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
    
    # Threshold the gradient to get clear boundaries
    _, boundaries = cv.threshold(gradient, 10, 255, cv.THRESH_BINARY)
    
    # Combine with transparency mask
    boundaries = boundaries & mask
    
    # If no boundaries found, try a different approach using color differences
    if np.count_nonzero(boundaries) == 0:
        # Create a color-difference based boundary detection
        # Convert RGB to a single channel representation for connected components
        # Use a simple hash of RGB values
        h, w = rgb.shape[:2]
        color_hash = (rgb[:,:,0].astype(np.uint32) * 65536 + 
                     rgb[:,:,1].astype(np.uint32) * 256 + 
                     rgb[:,:,2].astype(np.uint32))
        
        # Find connected components based on color hash
        num_labels, labels = cv.connectedComponents(color_hash.astype(np.uint8), connectivity=connectivity)
        
        # Create boundary image
        boundaries = np.zeros_like(gray)
        
        # For each region, find its boundary
        for label in range(1, num_labels):
            # Create mask for this region
            region_mask = (labels == label).astype(np.uint8)
            
            # Find contours of this region
            contours, _ = cv.findContours(region_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Draw contours as boundaries
            cv.drawContours(boundaries, contours, -1, 255, 1)
        
        # Combine with transparency mask
        boundaries = boundaries & mask
    
    # If still no boundaries, use very sensitive Canny as last resort
    if np.count_nonzero(boundaries) == 0:
        edges = cv.Canny(gray, 5, 15)  # Very low thresholds
        boundaries = edges & mask
    
    # Convert to RGBA format (white boundaries on transparent background)
    result = np.zeros((rgba.shape[0], rgba.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = 255  # White color
    result[:, :, 3] = boundaries  # Alpha channel
    
    return result


def smooth_region_boundaries(
    rgba: np.ndarray, 
    method: str = "morphological",
    strength: float = 0.5,
    preserve_colors: bool = True
) -> np.ndarray:
    """
    Smooth region boundaries using various techniques.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image
    method: str
        Smoothing method: "morphological", "bilateral", "contour", "gaussian", "multiscale"
    strength: float
        Smoothing strength from 0.0 (no smoothing) to 1.0 (strong smoothing)
    preserve_colors: bool
        Whether to preserve original palette colors
        
    Returns
    -------
    np.ndarray
        Smoothed RGBA image
    """
    if method == "morphological":
        return _morphological_smoothing(rgba, strength, preserve_colors)
    elif method == "bilateral":
        return _bilateral_smoothing(rgba, strength, preserve_colors)
    elif method == "contour":
        return _contour_smoothing(rgba, strength, preserve_colors)
    elif method == "gaussian":
        return _gaussian_smoothing(rgba, strength, preserve_colors)
    elif method == "multiscale":
        return _multiscale_smoothing(rgba, strength, preserve_colors)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def _morphological_smoothing(rgba: np.ndarray, strength: float, preserve_colors: bool) -> np.ndarray:
    """Morphological smoothing using opening and closing operations."""
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Only process non-transparent pixels
    non_transparent = alpha > 0
    if not np.any(non_transparent):
        return result
    
    # Calculate kernel size based on strength (1-5 pixels)
    kernel_size = max(1, int(strength * 4) + 1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Process each unique color separately to preserve palette
    unique_colors = np.unique(rgb[non_transparent].reshape(-1, 3), axis=0)
    
    # Create a temporary result to avoid overwriting during processing
    temp_result = np.zeros_like(rgba)
    
    for color in unique_colors:
        # Create mask for this color
        color_mask = np.all(rgb == color, axis=2) & non_transparent
        color_mask = color_mask.astype(np.uint8) * 255
        
        # Apply morphological operations
        # Opening: removes small protrusions
        opened = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel)
        # Closing: fills small holes
        closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
        
        # Update the temporary result for this color
        temp_result[closed > 0] = [color[0], color[1], color[2], 255]
    
    # Only update pixels that were originally non-transparent
    mask = temp_result[:, :, 3] > 0
    result[mask] = temp_result[mask]
    
    return result


def _bilateral_smoothing(rgba: np.ndarray, strength: float, preserve_colors: bool) -> np.ndarray:
    """Bilateral filtering for edge-preserving smoothing."""
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Only process non-transparent pixels
    non_transparent = alpha > 0
    if not np.any(non_transparent):
        return result
    
    # Calculate filter parameters based on strength
    d = max(1, int(strength * 9) + 1)  # Neighborhood diameter
    sigma_color = max(1, int(strength * 75) + 1)  # Color similarity
    sigma_space = max(1, int(strength * 75) + 1)  # Spatial similarity
    
    # Apply bilateral filter
    filtered = cv.bilateralFilter(rgb, d, sigma_color, sigma_space)
    
    # Preserve original colors if requested
    if preserve_colors:
        # Find closest palette colors
        unique_colors = np.unique(rgb[non_transparent].reshape(-1, 3), axis=0)
        for i in range(filtered.shape[0]):
            for j in range(filtered.shape[1]):
                if non_transparent[i, j]:
                    # Find closest original color
                    pixel_color = filtered[i, j]
                    distances = [np.linalg.norm(pixel_color - orig_color) for orig_color in unique_colors]
                    closest_idx = np.argmin(distances)
                    filtered[i, j] = unique_colors[closest_idx]
    
    result[:, :, :3] = filtered
    return result


def _contour_smoothing(rgba: np.ndarray, strength: float, preserve_colors: bool) -> np.ndarray:
    """Contour-based smoothing using contour approximation."""
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Only process non-transparent pixels
    non_transparent = alpha > 0
    if not np.any(non_transparent):
        return result
    
    # Create a mask for non-transparent pixels
    mask = non_transparent.astype(np.uint8) * 255
    
    # Find connected components for each unique color
    unique_colors = np.unique(rgb[non_transparent].reshape(-1, 3), axis=0)
    
    # Create new image - ensure it's contiguous
    smoothed = np.zeros_like(rgba)
    smoothed = np.ascontiguousarray(smoothed)
    
    for color in unique_colors:
        # Create mask for this color
        color_mask = np.all(rgb == color, axis=2) & non_transparent
        color_mask = color_mask.astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Smooth contours
            epsilon = strength * 0.02 * cv.arcLength(contours[0], True)
            smoothed_contours = [cv.approxPolyDP(contour, epsilon, True) for contour in contours]
            
            # Create a temporary mask for filling
            temp_mask = np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.uint8)
            
            # Fill contours in the mask
            for contour in smoothed_contours:
                cv.fillPoly(temp_mask, [contour], 255)
            
            # Apply the color to the smoothed result
            smoothed[temp_mask > 0, :3] = color
            smoothed[temp_mask > 0, 3] = 255
    
    return smoothed


def _gaussian_smoothing(rgba: np.ndarray, strength: float, preserve_colors: bool) -> np.ndarray:
    """Gaussian blur with better color preservation to avoid artifacts."""
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Only process non-transparent pixels
    non_transparent = alpha > 0
    if not np.any(non_transparent):
        return result
    
    # Calculate blur parameters - use more conservative values
    kernel_size = max(3, int(strength * 6) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd kernel size
    
    sigma = strength * 1.0  # More conservative sigma
    
    if preserve_colors:
        # Get unique colors
        unique_colors = np.unique(rgb[non_transparent].reshape(-1, 3), axis=0)
        
        # Create a result image
        smoothed_rgb = rgb.copy()
        
        # Process each color region separately
        for color in unique_colors:
            # Create mask for this color
            color_mask = np.all(rgb == color, axis=2) & non_transparent
            color_mask = color_mask.astype(np.uint8) * 255
            
            # Apply a small morphological close to smooth the mask edges
            small_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            smoothed_mask = cv.morphologyEx(color_mask, cv.MORPH_CLOSE, small_kernel)
            
            # Apply Gaussian blur to the mask only
            blurred_mask = cv.GaussianBlur(smoothed_mask, (kernel_size, kernel_size), sigma)
            
            # Create a temporary image with this color
            temp_image = np.zeros_like(rgb)
            temp_image[color_mask > 0] = color
            
            # Apply Gaussian blur to the temporary image
            blurred_temp = cv.GaussianBlur(temp_image, (kernel_size, kernel_size), sigma)
            
            # Use the blurred mask to blend the blurred color back
            mask_normalized = blurred_mask.astype(np.float32) / 255.0
            
            # Only update pixels where the mask is strong enough
            strong_mask = mask_normalized > 0.3
            
            for c in range(3):
                smoothed_rgb[strong_mask, c] = (
                    smoothed_rgb[strong_mask, c] * (1 - mask_normalized[strong_mask]) +
                    blurred_temp[strong_mask, c] * mask_normalized[strong_mask]
                ).astype(np.uint8)
        
        result[:, :, :3] = smoothed_rgb
    else:
        # Simple Gaussian blur without color preservation
        blurred = cv.GaussianBlur(rgb, (kernel_size, kernel_size), sigma)
        result[:, :, :3] = blurred
    
    return result


def _multiscale_smoothing(rgba: np.ndarray, strength: float, preserve_colors: bool) -> np.ndarray:
    """Multi-scale smoothing based on region size."""
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Only process non-transparent pixels
    non_transparent = alpha > 0
    if not np.any(non_transparent):
        return result
    
    # Analyze regions to determine sizes
    stats = analyze_regions(rgba, min_size_threshold=10)
    all_regions = stats.get('all_regions', [])
    
    if not all_regions:
        return result
    
    # Calculate size thresholds
    sizes = [r['size'] for r in all_regions]
    max_size = max(sizes)
    min_size = min(sizes)
    
    # Define size categories
    large_threshold = min_size + (max_size - min_size) * 0.7
    medium_threshold = min_size + (max_size - min_size) * 0.3
    
    # Create size-based smoothing
    smoothed = np.zeros_like(rgba)
    
    for region in all_regions:
        region_size = region['size']
        color = region['color']
        
        # Determine smoothing strength based on size
        if region_size >= large_threshold:
            region_strength = strength * 0.3  # Light smoothing for large regions
        elif region_size >= medium_threshold:
            region_strength = strength * 0.6  # Medium smoothing for medium regions
        else:
            region_strength = strength * 1.0  # Strong smoothing for small regions
        
        # Create mask for this region
        color_mask = np.all(rgb == color, axis=2) & non_transparent
        color_mask = color_mask.astype(np.uint8) * 255
        
        # Apply appropriate smoothing
        if region_strength > 0.1:
            kernel_size = max(1, int(region_strength * 5) + 1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Apply morphological smoothing
            smoothed_mask = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel)
            smoothed_mask = cv.morphologyEx(smoothed_mask, cv.MORPH_CLOSE, kernel)
        else:
            smoothed_mask = color_mask
        
        # Fill the smoothed region
        smoothed[smoothed_mask > 0] = [color[0], color[1], color[2], 255]
    
    return smoothed
