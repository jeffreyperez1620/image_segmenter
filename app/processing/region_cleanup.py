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
                    'centroid': (float(centroids[i][0]), float(centroids[i][1])),  # (x, y)
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


def find_neighboring_colors_for_component(rgba: np.ndarray, component_mask: np.ndarray, connectivity: int = 8) -> Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int, int], int], int, bool, np.ndarray]:
    """
    Find colors that are adjacent to a specific connected component, along with their counts.
    
    Parameters
    ----------
    rgba: np.ndarray
        Input RGBA image
    component_mask: np.ndarray
        Boolean mask for the specific component
    connectivity: int
        Connectivity (4 or 8) for neighbor detection
        
    Returns
    -------
    Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int, int], int], int, bool, np.ndarray]
        Tuple of (list of unique neighbor colors, dict of color -> count, total adjacent pixels, has_adjacent_pixels, adjacent_mask)
        has_adjacent_pixels indicates if any adjacent pixels exist (regardless of transparency)
        adjacent_mask is a boolean mask of pixels adjacent to the component (non-transparent only)
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
    
    # Check if adjacent pixels exist (regardless of transparency)
    has_adjacent_pixels = np.any((dilated_mask > 0) & ~component_mask)
    
    # Find pixels that are adjacent but not part of the component (only non-transparent ones)
    adjacent_mask = (dilated_mask > 0) & ~component_mask & (alpha > 0)
    
    # Get unique colors and their counts in adjacent areas
    if np.any(adjacent_mask):
        adjacent_pixels = rgb[adjacent_mask]
        unique_colors, counts = np.unique(adjacent_pixels.reshape(-1, 3), axis=0, return_counts=True)
        color_list = [tuple(color) for color in unique_colors]
        color_counts = {tuple(color): int(count) for color, count in zip(unique_colors, counts)}
        total_adjacent = len(adjacent_pixels)
        return (color_list, color_counts, total_adjacent, has_adjacent_pixels, adjacent_mask)
    else:
        return ([], {}, 0, has_adjacent_pixels, adjacent_mask)


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
    weights: dict = None,
    neighbor_centroid: Optional[Tuple[float, float]] = None
) -> float:
    """
    Calculate a score for merging a small region into a neighbor color.
    Optimized with spatial proximity calculation.
    
    Parameters
    ----------
    small_region: dict
        Information about the small region (must include 'centroid' for spatial calculation)
    neighbor_color: Tuple[int, int, int]
        RGB color of the potential merge target
    neighbor_region_size: int
        Size of the neighbor region in pixels
    image_context: dict
        Context information about the image
    weights: dict, optional
        Weights for different factors
    neighbor_centroid: Optional[Tuple[float, float]]
        Centroid of the neighbor region for spatial calculation
        
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
    if neighbor_centroid is not None and 'centroid' in small_region:
        # Calculate distance between centroids
        small_centroid = small_region['centroid']
        dx = small_centroid[0] - neighbor_centroid[0]
        dy = small_centroid[1] - neighbor_centroid[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Normalize by image diagonal (max possible distance)
        max_distance = image_context.get('max_distance', 1.0)
        if max_distance > 0:
            spatial_prox = 1.0 - min(1.0, distance / max_distance)
        else:
            spatial_prox = 0.5
    else:
        spatial_prox = 0.5  # Fallback if centroids not available
    
    # Factor 3: Frequency (0-1, higher is more common)
    total_pixels = image_context.get('total_pixels', 1)
    neighbor_pixel_count = image_context.get('color_counts', {}).get(neighbor_color, 1)
    frequency = min(1.0, neighbor_pixel_count / (total_pixels * 0.1))  # Normalize
    
    # Factor 4: Region size (prefer larger neighbors)
    max_region_size = image_context.get('max_region_size', 1)
    size_factor = min(1.0, neighbor_region_size / max_region_size) if max_region_size > 0 else 0.0
    
    # Weighted combination
    score = (weights['color'] * color_sim + 
             weights['spatial'] * spatial_prox + 
             weights['frequency'] * frequency + 
             weights['size'] * size_factor)
    
    return score


# Pre-computed LAB color cache for faster distance calculations
_lab_color_cache: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
_max_lab_distance = 255.0 * np.sqrt(3)  # Maximum possible distance in LAB space

def _rgb_to_lab_cached(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB with caching."""
    if rgb not in _lab_color_cache:
        import cv2 as cv
        img = np.array([[rgb]], dtype=np.uint8)
        lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        _lab_color_cache[rgb] = tuple(lab[0, 0].astype(np.float64))
    return _lab_color_cache[rgb]

def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Calculate perceptual color distance between two RGB colors.
    Returns a value between 0 and 1, where 0 is identical and 1 is maximally different.
    Optimized with LAB color space caching.
    """
    if color1 == color2:
        return 0.0
    
    # Use cached LAB conversions
    l1, a1, b1 = _rgb_to_lab_cached(color1)
    l2, a2, b2 = _rgb_to_lab_cached(color2)
    
    # Calculate Euclidean distance in LAB space
    l_diff = l1 - l2
    a_diff = a1 - a2
    b_diff = b1 - b2
    
    # Normalize to 0-1 range
    distance = np.sqrt(l_diff**2 + a_diff**2 + b_diff**2) / _max_lab_distance
    
    return min(1.0, max(0.0, distance))


def _build_region_graph(
    all_regions: List[dict],
    rgba: np.ndarray,
    connectivity: int = 8,
    progress_callback: Optional[callable] = None
) -> Tuple[Dict[int, dict], np.ndarray]:
    """
    Build a graph of regions with unique IDs and their neighbors.
    
    Parameters
    ----------
    all_regions: List[dict]
        List of region dictionaries from analyze_regions
    rgba: np.ndarray
        RGBA image
    connectivity: int
        Connectivity for neighbor detection (4 or 8)
        
    Returns
    -------
    Tuple[Dict[int, dict], np.ndarray]
        Tuple of (region_registry, global_region_labels)
        region_registry maps region_id -> {size, color, neighbor_ids, component_id, labels, centroid, bbox}
        global_region_labels maps each pixel to its region_id (0 = no region/transparent)
    """
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    
    # Create global region labels array (0 = transparent/no region)
    global_region_labels = np.zeros((h, w), dtype=np.int32)
    
    # Assign unique IDs to each region and build registry
    region_registry: Dict[int, dict] = {}
    total_regions = len(all_regions)
    region_id_counter = 0
    
    for i, region in enumerate(all_regions):
        if progress_callback:
            # First 50% of progress: assigning IDs and building registry
            progress_callback(int((i / total_regions) * 50), 100, f"Building region graph: Assigning IDs ({i+1}/{total_regions})...")
        
        labels = region['labels']
        component_id = region['component_id']
        component_mask = (labels == component_id) & (alpha > 0)
        
        # Mark pixels in global labels array
        global_region_labels[component_mask] = region_id_counter
        
        # Store region info
        region_registry[region_id_counter] = {
            'size': region['size'],
            'color': region['color'],
            'neighbor_ids': set(),
            'component_id': component_id,
            'labels': labels,
            'centroid': region['centroid'],
            'bbox': region['bbox']
        }
        region_id_counter += 1

    # Build neighbor graph using dilation
    if connectivity == 4:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        kernel = np.ones((3, 3), np.uint8)
    
    region_items = list(region_registry.items())
    total_for_neighbors = len(region_items)
    
    for idx, (region_id, region_data) in enumerate(region_items):
        if progress_callback:
            # Second 50% of progress: building neighbor relationships
            progress = 50 + int((idx / total_for_neighbors) * 50)
            progress_callback(progress, 100, f"Building region graph: Finding neighbors ({idx+1}/{total_for_neighbors})...")
        
        labels = region_data['labels']
        component_id = region_data['component_id']
        component_mask = (labels == component_id) & (alpha > 0).astype(np.uint8)
        
        # Dilate to find adjacent pixels
        dilated_mask = cv.dilate(component_mask, kernel, iterations=1)
        adjacent_mask = (dilated_mask > 0) & (component_mask == 0) & (alpha > 0)
        
        # Find which region IDs are adjacent
        adjacent_coords = np.where(adjacent_mask)
        if len(adjacent_coords[0]) > 0:
            adjacent_region_ids = global_region_labels[adjacent_coords]

            # Convert numpy int32 to Python int to avoid type issues
            # Filter out zeros (transparent/no region) and the region's own ID
            # The region's own ID can appear in adjacent positions for C-shaped or concave regions
            # where dilation causes the dilated contour to include pixels labeled as this region
            unique_neighbor_ids = set(int(nid) for nid in adjacent_region_ids[adjacent_region_ids > 0] if int(nid) != region_id)
            
            region_data['neighbor_ids'] = unique_neighbor_ids
            
            # Make the relationship symmetric: if A is a neighbor of B, then B is also a neighbor of A
            # This should naturally be symmetric, but we enforce it to catch any bugs in neighbor detection
            # Since neighbor_ids is a set, .add() is idempotent - no need to check if it already exists
            for neighbor_id in unique_neighbor_ids:
                if neighbor_id in region_registry:
                    region_registry[neighbor_id]['neighbor_ids'].add(region_id)
                else:
                    # Neighbor hasn't been processed yet or was deleted (shouldn't happen during initialization)
                    print(f"WARNING: Region {neighbor_id} not found in registry when making relationship symmetric for region {region_id}")
    
    return region_registry, global_region_labels


def _merge_regions_in_graph(
    region_registry: Dict[int, dict],
    region_a_id: int,
    region_b_id: int,
    rgba: np.ndarray,
    small_region_ids: set,
    min_size: int,
    image_context: dict
) -> None:
    """
    Merge region A into region B in the graph and update the image.
    
    Parameters
    ----------
    region_registry: Dict[int, dict]
        Registry of all regions
    region_a_id: int
        ID of region to merge (will be removed)
    region_b_id: int
        ID of region to merge into (will be updated)
    rgba: np.ndarray
        Image to update
    small_region_ids: set
        Set of small region IDs to maintain
    min_size: int
        Minimum size threshold for small regions
    """
    # Ensure IDs are Python ints (not numpy int32)
    region_a_id = int(region_a_id)
    region_b_id = int(region_b_id)
    
    # Safety check: don't merge a region into itself
    if region_a_id == region_b_id:
        error_msg = f"Cannot merge region {region_a_id} into itself"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    # Safety check: ensure both regions exist
    if region_a_id not in region_registry:
        error_msg = f"Region {region_a_id} not found in registry (may have been deleted)"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    if region_b_id not in region_registry:
        error_msg = f"Region {region_b_id} not found in registry (may have been deleted)"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    region_a = region_registry[region_a_id]
    region_b = region_registry[region_b_id]
    
    # Update color_counts: remove A's pixels, B's count will be updated with new size
    color_a = region_a['color']
    color_b = region_b['color']
    if color_a in image_context['color_counts']:
        image_context['color_counts'][color_a] -= region_a['size']
        if image_context['color_counts'][color_a] <= 0:
            del image_context['color_counts'][color_a]
    
    # Update B's size
    region_b['size'] += region_a['size']
    
    # Update color_counts: B's color count increases by A's size
    if color_b not in image_context['color_counts']:
        image_context['color_counts'][color_b] = 0
    image_context['color_counts'][color_b] += region_a['size']
    
    # Update max_region_size if B is now larger
    if region_b['size'] > image_context['max_region_size']:
        image_context['max_region_size'] = region_b['size']
    
    # Update small_region_ids: remove A (being deleted) and B if it's no longer small
    small_region_ids.discard(region_a_id)
    if region_b_id in small_region_ids and region_b['size'] >= min_size:
        small_region_ids.discard(region_b_id)
    
    # Update neighbors of A (except B)
    # Create a copy to avoid "set changed size during iteration" error
    for neighbor_id in list(region_a['neighbor_ids']):
        if neighbor_id != region_b_id and neighbor_id in region_registry:
            neighbor = region_registry[neighbor_id]
            neighbor['neighbor_ids'].discard(region_a_id)
            neighbor['neighbor_ids'].add(region_b_id)
            # Safety check: ensure neighbor doesn't have itself as a neighbor
            if neighbor_id in neighbor['neighbor_ids']:
                error_msg = f"ERROR: After updating neighbor {neighbor_id} to point to {region_b_id}, neighbor {neighbor_id} has itself in neighbor_ids! Removing self-reference."  
                print(error_msg)
                raise RuntimeError(error_msg)
    
    # Update B's neighbors: union of A's and B's neighbors, minus A and B
    region_b['neighbor_ids'] = (region_a['neighbor_ids'] | region_b['neighbor_ids']) - {region_a_id, region_b_id}
    # Check for neighbors that no longer exist in the registry (this indicates a bug)
    deleted_neighbors = {nid for nid in region_b['neighbor_ids'] if nid not in region_registry}
    if deleted_neighbors:
        error_msg = (
            f"ERROR: After merging {region_a_id} into {region_b_id}, region {region_b_id} has neighbors "
            f"that no longer exist in registry: {deleted_neighbors}. This indicates a bug in graph maintenance."
        )
        print(error_msg)
        raise RuntimeError(error_msg)
    # Safety check: ensure B doesn't have itself as a neighbor
    if region_b_id in region_b['neighbor_ids']:
        error_msg = f"ERROR: After merging {region_a_id} into {region_b_id}, region {region_b_id} has itself in neighbor_ids!"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Update image: set A's pixels to B's color
    labels_a = region_a['labels']
    component_id_a = region_a['component_id']
    alpha = rgba[:, :, 3]
    component_mask_a = (labels_a == component_id_a) & (alpha > 0)
    
    rgb = rgba[:, :, :3]
    rgb[component_mask_a] = region_b['color']
    
    # Remove A from registry
    del region_registry[region_a_id]
    
    # Recursive merge: After merging A into B, check if B now has new neighbors of the same color
    # This handles the corner case where a small region was sandwiched between two regions
    # of the same color (e.g., red-green-red). After merging green into red, the two red
    # regions are now adjacent and should be merged.
    # Use an open set (queue) to track neighbors that need to be checked/merged
    # This allows us to add new neighbors discovered during merging without iteration issues
    # Check for neighbors that no longer exist in the registry (this indicates a bug)
    deleted_neighbors = {nid for nid in region_b['neighbor_ids'] if nid not in region_registry}
    if deleted_neighbors:
        error_msg = (
            f"ERROR: Before recursive merge, region {region_b_id} has neighbors that no longer exist "
            f"in registry: {deleted_neighbors}. This indicates a bug in graph maintenance."
        )
        print(error_msg)
        raise RuntimeError(error_msg)
    same_color_neighbors_to_check = set(region_b['neighbor_ids'])

    # Process the queue until empty
    while same_color_neighbors_to_check:
        # Get the next neighbor to merge
        c_id = same_color_neighbors_to_check.pop()
        # Double-check it still exists (it might have been merged/deleted)
        if c_id not in region_registry:
            print(f"ERROR: Neighbor {c_id} not found in registry (may have been merged/deleted)")
            continue
        region_c = region_registry[c_id]
        if region_c['color'] != region_b['color']:
            same_color_neighbors_to_check.discard(c_id)
            continue
        # Neighbor is the same color as B, merge it into B
        # First, clean up C's neighbor list to remove any deleted regions (safety check)
        # This can happen if C's neighbor list wasn't properly updated when a previous merge occurred
        deleted_in_c = {nid for nid in region_c['neighbor_ids'] if nid not in region_registry}
        if deleted_in_c:
            error_msg = (
                f"ERROR: Region {c_id} has neighbors that no longer exist in registry: {deleted_in_c}. "
                f"This indicates a bug - C's neighbor list should have been updated when those regions were merged/deleted."
            )
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Update neighbors of C (except B) to point to B instead
        for c_neighbor_id in list(region_c['neighbor_ids']):
            if c_neighbor_id != region_b_id and c_neighbor_id in region_registry:
                region_c_neighbor = region_registry[c_neighbor_id] 
                region_c_neighbor['neighbor_ids'].discard(c_id)
                region_c_neighbor['neighbor_ids'].add(region_b_id)
                # Safety check: ensure neighbor doesn't have itself as a neighbor
                if c_neighbor_id in region_c_neighbor['neighbor_ids']:
                    error_msg = f"ERROR: After updating neighbor {c_neighbor_id} to point to {region_b_id} (recursive merge), neighbor {c_neighbor_id} has itself in neighbor_ids!"
                    print(error_msg)
                    raise RuntimeError(error_msg)
        
        # Update B's neighbors: union of C's and B's neighbors, minus C and B
        # This is critical to prevent B from having itself in its neighbor list
        # Note: We've already verified that C's neighbor list doesn't contain deleted regions
        region_b['neighbor_ids'] = (region_c['neighbor_ids'] | region_b['neighbor_ids']) - {c_id, region_b_id}
        # Check for neighbors that no longer exist in the registry (this indicates a bug)
        deleted_neighbors = {nid for nid in region_b['neighbor_ids'] if nid not in region_registry}
        if deleted_neighbors:
            error_msg = (
                f"ERROR: After recursively merging {c_id} into {region_b_id}, region {region_b_id} has neighbors "
                f"that no longer exist in registry: {deleted_neighbors}. This indicates a bug in graph maintenance."
            )
            print(error_msg)
            raise RuntimeError(error_msg)
        # Safety check: ensure B doesn't have itself as a neighbor
        if region_b_id in region_b['neighbor_ids']:
            error_msg = f"ERROR: After recursively merging {c_id} into {region_b_id}, region {region_b_id} has itself in neighbor_ids!"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        region_b['size'] += region_c['size']
        if region_b['size'] > image_context['max_region_size']:
            image_context['max_region_size'] = region_b['size']
        if region_b_id in small_region_ids and region_b['size'] >= min_size:
            small_region_ids.discard(region_b_id)
        del region_registry[c_id]
        
        same_color_neighbors_to_check.discard(c_id)


def merge_small_regions(
    rgba: np.ndarray, 
    min_size: int, 
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
    merge_weights: dict, optional
        Weights for different merge factors in scoring
    progress_callback: callable, optional
        Callback function for progress updates (current, total, message)
    connectivity: int
        Connectivity for neighbor detection (4 or 8)
        
    Returns
    -------
    np.ndarray
        Image with small regions merged
    """
    result = rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Clear LAB color cache at start to prevent memory buildup
    _lab_color_cache.clear()
    
    # Analyze regions with the minimum size threshold
    if progress_callback:
        progress_callback(0, 100, "Analyzing regions...")
    
    stats = analyze_regions(rgba, min_size, connectivity)
    all_regions = stats.get('all_regions', [])
    
    if progress_callback:
        progress_callback(10, 100, "Building image context...")
    
    # Calculate image diagonal for spatial proximity normalization
    h, w = rgba.shape[:2]
    image_diagonal = np.sqrt(h*h + w*w)
    
    # Initial image context (will be updated incrementally)
    # Calculate initial max_region_size and color_counts from all_regions
    max_region_size = max([r['size'] for r in all_regions]) if all_regions else 1
    color_counts = {}
    for region in all_regions:
        color = region['color']
        if color not in color_counts:
            color_counts[color] = 0
        color_counts[color] += region['size']
    
    image_context = {
        'total_pixels': np.sum(alpha > 0),
        'max_region_size': max_region_size,
        'max_distance': image_diagonal,
        'color_counts': color_counts
    }
    
    # Build region graph once at the start
    # Progress range: 15-20 (out of 100 total)
    def graph_progress_callback(current: int, total: int, message: str) -> None:
        if progress_callback:
            # Map 0-100 from graph building to 15-20 in overall progress
            overall_progress = 15 + int((current / total) * 5)
            progress_callback(overall_progress, 100, message)
    
    region_registry, global_region_labels = _build_region_graph(all_regions, result, connectivity, graph_progress_callback)
    
    # Build initial set of small regions (regions below threshold)
    # This set will be maintained incrementally - regions only get larger or are removed
    small_region_ids = set()
    for rid, rdata in region_registry.items():
        if rdata['size'] < min_size:
            small_region_ids.add(rid)
    
    # Perform multiple passes to ensure all small regions are handled
    max_passes = 20  # Increased to allow complete cleanup
    total_merged = 0
    previous_small_count = float('inf')  # Track progress to detect when no more progress is possible
    
    for pass_num in range(1, max_passes + 1):
        if not small_region_ids:
            # No more small regions found
            break
        
        # Check if we're making progress
        current_small_count = len(small_region_ids)
        if current_small_count >= previous_small_count:
            # No progress made - stop to prevent infinite loop
            print(f"No progress made in pass {pass_num}, stopping early")
            break
        previous_small_count = current_small_count
        
        if progress_callback:
            progress_callback(20 + (pass_num - 1) * 25, 100, f"Pass {pass_num}: Found {len(small_region_ids)} small regions to process...")
        
        merged_count = 0
        total_regions = len(small_region_ids)
        processed_count = 0
        
        # Process regions by popping from the set to avoid iteration issues
        while small_region_ids:
            # Pop a region ID from the set
            region_id = small_region_ids.pop()
            
            # Update progress
            processed_count += 1
            if progress_callback:
                progress = 20 + (pass_num - 1) * 25 + int((processed_count / total_regions) * 20)  # 20-90% for processing
                progress_callback(progress, 100, f"Pass {pass_num}: Processing region {processed_count}/{total_regions}...")
            
            # Get region data from graph
            if region_id not in region_registry:
                continue
            
            region_data = region_registry[region_id]
            target_color = region_data['color']
            region_size = region_data['size']
            labels = region_data['labels']
            component_id = region_data['component_id']
            
            # Create mask for this specific connected component
            component_mask = (labels == component_id) & (alpha > 0)
            
            # Check if region has no valid pixels in mask
            if not np.any(component_mask):
                continue
            
            # Get neighbors from graph
            neighbor_ids = region_data['neighbor_ids']
            
            # Safety check: ensure region doesn't have itself as a neighbor
            if region_id in neighbor_ids:
                error_msg = f"ERROR: Region {region_id} has itself in neighbor_ids before processing! neighbor_ids: {neighbor_ids}"
                print(error_msg)
                raise RuntimeError(error_msg)
            
            # Check if region has no neighbors (surrounded by transparent or at edge)
            if not neighbor_ids:
                # No neighbors - delete the region
                alpha[component_mask] = 0
                merged_count += 1
                # Update color_counts: remove this region's pixels
                if target_color in image_context['color_counts']:
                    image_context['color_counts'][target_color] -= region_size
                    if image_context['color_counts'][target_color] <= 0:
                        del image_context['color_counts'][target_color]
                # Remove from registry and small_region_ids
                small_region_ids.discard(region_id)
                del region_registry[region_id]
                continue
            
            # Get neighbor region data and group by color for scoring
            # Create a copy to avoid "set changed size during iteration" error
            neighbor_by_color: Dict[Tuple[int, int, int], List[Tuple[int, dict]]] = {}
            
            for neighbor_id in list(neighbor_ids):
                if neighbor_id not in region_registry:
                    # ERROR: Neighbor was removed from registry but is still in neighbor_ids set
                    # This indicates a bug in the graph maintenance - _merge_regions_in_graph should
                    # have updated our neighbor_ids when the neighbor was merged.
                    error_msg = (
                        f"Graph consistency error: Region {region_id} has neighbor {neighbor_id} "
                        f"in neighbor_ids set, but {neighbor_id} is not in registry. "
                        f"This indicates a bug in the merge function - it should have updated "
                        f"neighbor_ids when the neighbor was merged."
                    )
                    print(f"ERROR: {error_msg}")
                    raise RuntimeError(error_msg)
                neighbor_data = region_registry[neighbor_id]
                neighbor_color = neighbor_data['color']
                if neighbor_color not in neighbor_by_color:
                    neighbor_by_color[neighbor_color] = []
                neighbor_by_color[neighbor_color].append((neighbor_id, neighbor_data))
            
            # At this point, neighbor_by_color should not be empty because:
            # 1. We already checked if neighbor_ids is empty (line 664) and handled that case
            # 2. We raise an error if any neighbor_id is invalid (line 695)
            # 3. So if we reach here, all neighbor_ids are valid and neighbor_by_color has entries
            if not neighbor_by_color:
                error_msg = (
                    f"Graph consistency error: Region {region_id} has neighbors in neighbor_ids set, "
                    f"but neighbor_by_color is empty after processing. This should not be possible."
                )
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Determine which neighbor to merge to
            if len(neighbor_by_color) == 1:
                # Single neighbor color - merge to that color
                # Since all neighbors have the same color, it doesn't matter which one we pick
                neighbor_color = list(neighbor_by_color.keys())[0]
                neighbor_list = neighbor_by_color[neighbor_color]
                best_neighbor_id, best_neighbor = neighbor_list[0]
            else:
                # Multiple neighbor colors - use scores to select the best one
                neighbor_scores = []
                for neighbor_color, neighbor_list in neighbor_by_color.items():
                    # neighbor_list is List[Tuple[int, dict]] - find largest by size
                    best_neighbor_id_for_color, best_neighbor = max(neighbor_list, key=lambda x: x[1]['size'])
                    score = calculate_merge_score(
                        region_data,
                        neighbor_color,
                        best_neighbor['size'],
                        image_context,
                        merge_weights,
                        best_neighbor['centroid']
                    )
                    neighbor_scores.append((neighbor_color, best_neighbor_id_for_color, best_neighbor, score))
                
                # Check if we have any valid scores
                if not neighbor_scores:
                    continue
                
                # Sort by score (highest first) and select best
                neighbor_scores.sort(key=lambda x: x[3], reverse=True)
                best_color, best_neighbor_id, best_neighbor, best_score = neighbor_scores[0]
            
            if best_neighbor_id is None:
                continue
            
            # Merge region into best neighbor
            _merge_regions_in_graph(region_registry, region_id, best_neighbor_id, result, small_region_ids, min_size, image_context)
            merged_count += 1
            
            # Update alpha reference after merge (image may have changed)
            alpha = result[:, :, 3]
        
        # Update totals for this pass
        total_merged += merged_count
        
        if progress_callback:
            progress_callback(20 + pass_num * 25, 100, f"Pass {pass_num} complete: {merged_count} regions merged")
    
    if progress_callback:
        progress_callback(100, 100, f"Complete: {total_merged} regions merged in {pass_num} passes")
    
    print(f"Region cleanup complete: {total_merged} regions merged in {pass_num} passes")
    
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
    # Create a proper copy of the RGB data for OpenCV
    rgb = result[:, :, :3].copy()
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
        # Copy the modified RGB data back to the result array
        result[:, :, :3] = rgb
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
