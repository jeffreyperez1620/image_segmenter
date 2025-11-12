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
    
    # Extract RGB and alpha channels
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    
    # Analyze both non-transparent and transparent pixels
    non_transparent = alpha > 0
    transparent = alpha == 0
    
    all_regions = []
    region_colors = []
    region_sizes = []
    small_regions_count = 0
    
    # First, process non-transparent regions (colored regions)
    if np.any(non_transparent):
        # Find connected components for each unique color
        unique_colors = np.unique(rgb[non_transparent].reshape(-1, 3), axis=0)
        
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
                        'color': tuple(int(c) for c in color),  # Convert numpy types to Python ints
                        'size': int(area),
                        'label': i,
                        'labels': labels,
                        'component_id': i,
                        'centroid': (float(centroids[i][0]), float(centroids[i][1])),  # (x, y)
                        'bbox': (
                            stats[i, cv.CC_STAT_LEFT],
                            stats[i, cv.CC_STAT_TOP], 
                            stats[i, cv.CC_STAT_WIDTH],
                            stats[i, cv.CC_STAT_HEIGHT]
                        ),
                        'is_transparent': False
                    }
                    all_regions.append(region_info)
                    region_colors.append(tuple(int(c) for c in color))
                    region_sizes.append(int(area))
                    
                    if area < min_size_threshold:
                        small_regions_count += 1
    
    # Second, process transparent regions (holes that can be filled)
    if np.any(transparent):
        # Create binary mask for transparent pixels
        transparent_mask = transparent.astype(np.uint8) * 255
        
        # Find connected components of transparent pixels
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(transparent_mask, connectivity=connectivity)
        
        # Process each transparent component (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            if area > 0:  # Only include non-empty regions
                region_info = {
                    'color': None,  # Special marker for transparent regions
                    'size': int(area),
                    'label': i,
                    'labels': labels,
                    'component_id': i,
                    'centroid': (float(centroids[i][0]), float(centroids[i][1])),  # (x, y)
                    'bbox': (
                        stats[i, cv.CC_STAT_LEFT],
                        stats[i, cv.CC_STAT_TOP], 
                        stats[i, cv.CC_STAT_WIDTH],
                        stats[i, cv.CC_STAT_HEIGHT]
                    ),
                    'is_transparent': True
                }
                all_regions.append(region_info)
                region_colors.append(None)  # Transparent regions have no color
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
    # For transparent regions (color is None), give maximum score since we want to fill them
    small_region_color = small_region.get('color')
    if small_region_color is None:
        # Transparent region - give maximum color similarity score to encourage merging
        color_sim = 1.0
    else:
        color_sim = 1.0 - color_distance(small_region_color, neighbor_color)
    
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
    # Handle None colors (transparent regions)
    if color1 is None or color2 is None:
        # If either color is None, return maximum distance
        return 1.0
    
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


def _get_component_mask(
    labels: np.ndarray,
    component_id: int,
    alpha: np.ndarray,
    is_transparent: bool
) -> np.ndarray:
    """
    Get full-size component mask for a region based on transparency.
    Used for neighbor detection and global label marking.
    
    Parameters
    ----------
    labels: np.ndarray
        Labels array from connected components
    component_id: int
        Component ID to extract
    alpha: np.ndarray
        Alpha channel of the image
    is_transparent: bool
        Whether the region is transparent
        
    Returns
    -------
    np.ndarray
        Boolean mask for the component (full image size)
    """
    if is_transparent:
        return (labels == component_id) & (alpha == 0)
    else:
        return (labels == component_id) & (alpha > 0)


def _get_component_mask_bbox(
    labels: np.ndarray,
    component_id: int,
    alpha: np.ndarray,
    is_transparent: bool,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Get bounding-box-sized component mask for a region.
    Used for caching region data to save memory.
    
    Parameters
    ----------
    labels: np.ndarray
        Labels array from connected components
    component_id: int
        Component ID to extract
    alpha: np.ndarray
        Alpha channel of the image
    is_transparent: bool
        Whether the region is transparent
    bbox: Tuple[int, int, int, int]
        Bounding box (x, y, width, height)
        
    Returns
    -------
    np.ndarray
        Boolean mask for the component in bounding box coordinates
    """
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height
    
    # Extract bounding box region from labels and alpha
    labels_bbox = labels[y_min:y_max, x_min:x_max]
    alpha_bbox = alpha[y_min:y_max, x_min:x_max]
    
    # Create mask in bbox coordinates
    if is_transparent:
        return (labels_bbox == component_id) & (alpha_bbox == 0)
    else:
        return (labels_bbox == component_id) & (alpha_bbox > 0)


def _update_image_for_merge(
    rgba: np.ndarray,
    labels: np.ndarray,
    component_id: int,
    is_source_transparent: bool,
    is_target_transparent: bool,
    target_color: Optional[Tuple[int, int, int]]
) -> None:
    """
    Update image pixels when merging a region.
    
    Parameters
    ----------
    rgba: np.ndarray
        Image to update (modified in place)
    labels: np.ndarray
        Labels array for the source region
    component_id: int
        Component ID of the source region
    is_source_transparent: bool
        Whether the source region is transparent
    is_target_transparent: bool
        Whether the target region is transparent
    target_color: Optional[Tuple[int, int, int]]
        Color of the target region (None if transparent)
    """
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    
    # Get component mask
    component_mask = _get_component_mask(labels, component_id, alpha, is_source_transparent)
    
    # Merge logic: transparent regions get filled, non-transparent regions get recolored
    if is_source_transparent and not is_target_transparent and target_color is not None:
        # Transparent region: fill with target color
        rgb[component_mask] = target_color
        alpha[component_mask] = 255
    elif not is_source_transparent:
        if not is_target_transparent and target_color is not None:
            # Non-transparent to non-transparent: change color
            rgb[component_mask] = target_color
        elif is_target_transparent:
            # Non-transparent to transparent: make transparent
            alpha[component_mask] = 0


def _delete_region_from_image(
    rgba: np.ndarray,
    labels: np.ndarray,
    component_id: int,
    is_transparent: bool
) -> None:
    """
    Delete a region by making its pixels transparent.
    
    Parameters
    ----------
    rgba: np.ndarray
        Image to update (modified in place)
    labels: np.ndarray
        Labels array for the region
    component_id: int
        Component ID of the region
    is_transparent: bool
        Whether the region is already transparent
    """
    if is_transparent:
        # Already transparent, nothing to do
        return
    
    alpha = rgba[:, :, 3]
    component_mask = _get_component_mask(labels, component_id, alpha, is_transparent)
    alpha[component_mask] = 0


def _update_color_counts_for_deletion(
    image_context: dict,
    color: Optional[Tuple[int, int, int]],
    size: int,
    is_transparent: bool
) -> None:
    """
    Update color_counts when a region is deleted.
    
    Parameters
    ----------
    image_context: dict
        Image context containing color_counts
    color: Optional[Tuple[int, int, int]]
        Color of the region being deleted
    size: int
        Size of the region being deleted
    is_transparent: bool
        Whether the region is transparent
    """
    if not is_transparent and color is not None and color in image_context['color_counts']:
        image_context['color_counts'][color] -= size
        if image_context['color_counts'][color] <= 0:
            del image_context['color_counts'][color]


def _update_color_counts_for_merge(
    image_context: dict,
    source_color: Optional[Tuple[int, int, int]],
    source_size: int,
    source_is_transparent: bool,
    target_color: Optional[Tuple[int, int, int]],
    target_is_transparent: bool
) -> None:
    """
    Update color_counts when a region is merged into another.
    
    Parameters
    ----------
    image_context: dict
        Image context containing color_counts
    source_color: Optional[Tuple[int, int, int]]
        Color of the source region being merged
    source_size: int
        Size of the source region
    source_is_transparent: bool
        Whether the source region is transparent
    target_color: Optional[Tuple[int, int, int]]
        Color of the target region
    target_is_transparent: bool
        Whether the target region is transparent
    """
    # Remove source region's pixels from color_counts
    if not source_is_transparent and source_color is not None and source_color in image_context['color_counts']:
        image_context['color_counts'][source_color] -= source_size
        if image_context['color_counts'][source_color] <= 0:
            del image_context['color_counts'][source_color]
    
    # Add source region's pixels to target region's color count
    if not target_is_transparent and target_color is not None:
        if target_color not in image_context['color_counts']:
            image_context['color_counts'][target_color] = 0
        image_context['color_counts'][target_color] += source_size


def _build_region_graph(
    all_regions: List[dict],
    rgba: np.ndarray,
    connectivity: int = 8,
    progress_callback: Optional[callable] = None
) -> Tuple[Dict[int, dict], np.ndarray, Dict[int, dict]]:
    """
    Build a graph of regions with unique IDs and their neighbors.
    Also creates a region cache with bounding-box masks for final rendering.
    
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
    Tuple[Dict[int, dict], np.ndarray, Dict[int, dict]]
        Tuple of (region_registry, global_region_labels, region_cache)
        region_registry maps region_id -> {size, color, neighbor_ids, contained_ids, component_id, labels, centroid, bbox, is_transparent}
        global_region_labels maps each pixel to its region_id (0 = no region/transparent)
        region_cache maps region_id -> {mask, color, bbox} for final rendering
    """
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    
    # Create global region labels array (0 = transparent/no region)
    global_region_labels = np.zeros((h, w), dtype=np.int32)
    
    # Assign unique IDs to each region and build registry
    region_registry: Dict[int, dict] = {}
    region_cache: Dict[int, dict] = {}  # Cache for final rendering: {mask, color, bbox}
    total_regions = len(all_regions)
    region_id_counter = 0
    
    for i, region in enumerate(all_regions):
        if progress_callback:
            # First 50% of progress: assigning IDs and building registry
            progress_callback(int((i / total_regions) * 50), 100, f"Building region graph: Assigning IDs ({i+1}/{total_regions})...")
        
        labels = region['labels']
        component_id = region['component_id']
        is_transparent = region.get('is_transparent', False)
        bbox = region['bbox']
        
        # Get full mask for global label marking
        component_mask = _get_component_mask(labels, component_id, alpha, is_transparent)
        
        # Mark pixels in global labels array
        global_region_labels[component_mask] = region_id_counter
        
        # Get bounding-box mask for caching
        bbox_mask = _get_component_mask_bbox(labels, component_id, alpha, is_transparent, bbox)
        
        # Store region info in registry
        region_registry[region_id_counter] = {
            'size': region['size'],
            'color': region['color'],
            'neighbor_ids': set(),
            'contained_ids': {region_id_counter},  # Initially contains only itself
            'component_id': component_id,
            'labels': labels,
            'centroid': region['centroid'],
            'bbox': bbox,
            'is_transparent': is_transparent
        }
        
        # Store region data in cache for final rendering
        region_cache[region_id_counter] = {
            'mask': bbox_mask,
            'color': region['color'],
            'bbox': bbox
        }
        
        region_id_counter += 1

    # Build neighbor graph using dilation to find adjacent pixels
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
        is_transparent = region_data.get('is_transparent', False)
        
        # Get component mask and dilate to find adjacent pixels
        component_mask = _get_component_mask(labels, component_id, alpha, is_transparent)
        component_mask_uint8 = component_mask.astype(np.uint8) * 255
        dilated_mask = cv.dilate(component_mask_uint8, kernel, iterations=1)
        boundary_mask = (dilated_mask > 0) & (component_mask_uint8 == 0)
        
        if is_transparent:
            # For transparent regions, adjacent pixels should be non-transparent (alpha > 0)
            adjacent_mask = boundary_mask & (alpha > 0)
        else:
            # For non-transparent regions, adjacent pixels can be transparent or non-transparent (any alpha)
            adjacent_mask = boundary_mask
        
        # Find which region IDs are adjacent - use vectorized operations
        if np.any(adjacent_mask):
            adjacent_region_ids = global_region_labels[adjacent_mask]
            # Use numpy unique for faster processing
            unique_neighbor_ids_array = np.unique(adjacent_region_ids[adjacent_region_ids > 0])
            # Filter out the region's own ID and convert to set
            unique_neighbor_ids = {int(nid) for nid in unique_neighbor_ids_array if int(nid) != region_id}
        else:
            unique_neighbor_ids = set()
        
        region_data['neighbor_ids'] = unique_neighbor_ids
        
        # Make the relationship symmetric: if A is a neighbor of B, then B is also a neighbor of A
        for neighbor_id in unique_neighbor_ids:
            if neighbor_id in region_registry:
                region_registry[neighbor_id]['neighbor_ids'].add(region_id)
            else:
                # Neighbor hasn't been processed yet or was deleted (shouldn't happen during initialization)
                print(f"WARNING: Region {neighbor_id} not found in registry when making relationship symmetric for region {region_id}")
    
    return region_registry, global_region_labels, region_cache


def _get_valid_merge_neighbors(
    region_id: int,
    region_data: dict,
    region_registry: Dict[int, dict]
) -> Dict[Tuple[int, int, int], List[Tuple[int, dict]]]:
    """
    Get valid neighbors for merging, filtering out transparent neighbors.
    
    Parameters
    ----------
    region_id: int
        ID of the region looking for neighbors
    region_data: dict
        Region data dictionary
    region_registry: Dict[int, dict]
        Registry of all regions
        
    Returns
    -------
    Dict[Tuple[int, int, int], List[Tuple[int, dict]]]
        Dictionary mapping neighbor colors to lists of (neighbor_id, neighbor_data) tuples
    """
    neighbor_ids = region_data['neighbor_ids']
    neighbor_by_color: Dict[Tuple[int, int, int], List[Tuple[int, dict]]] = {}
    
    for neighbor_id in list(neighbor_ids):
        if neighbor_id not in region_registry:
            # Neighbor was removed from registry but is still in neighbor_ids set
            # This indicates a bug in the graph maintenance
            error_msg = (
                f"Graph consistency error: Region {region_id} has neighbor {neighbor_id} "
                f"in neighbor_ids set, but {neighbor_id} is not in registry. "
                f"This indicates a bug in the merge function - it should have updated "
                f"neighbor_ids when the neighbor was merged."
            )
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        
        neighbor_data = region_registry[neighbor_id]
        neighbor_is_transparent = neighbor_data.get('is_transparent', False)
        
        # Skip transparent neighbors - we can't merge into transparent regions
        if neighbor_is_transparent:
            continue
        
        neighbor_color = neighbor_data['color']
        # Safety check: ensure neighbor_color is not None
        if neighbor_color is None:
            continue
        
        if neighbor_color not in neighbor_by_color:
            neighbor_by_color[neighbor_color] = []
        neighbor_by_color[neighbor_color].append((neighbor_id, neighbor_data))
    
    return neighbor_by_color


def _merge_regions_in_graph(
    region_registry: Dict[int, dict],
    region_a_id: int,
    region_b_id: int,
    small_region_ids: set,
    min_size: int,
    image_context: dict
) -> None:
    """
    Merge region A into region B in the graph.
    Updates containment lists instead of updating the image.
    
    Parameters
    ----------
    region_registry: Dict[int, dict]
        Registry of all regions
    region_a_id: int
        ID of region to merge (will be removed)
    region_b_id: int
        ID of region to merge into (will be updated)
    small_region_ids: set
        Set of small region IDs to maintain
    min_size: int
        Minimum size threshold for small regions
    image_context: dict
        Image context containing color_counts and max_region_size
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
    
    is_a_transparent = region_a.get('is_transparent', False)
    is_b_transparent = region_b.get('is_transparent', False)
    
    # Update color_counts for the merge
    _update_color_counts_for_merge(
        image_context,
        region_a['color'],
        region_a['size'],
        is_a_transparent,
        region_b['color'],
        is_b_transparent
    )
    
    # Update B's size
    region_b['size'] += region_a['size']
    
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
    
    # Update containment list: B now contains all regions that A contained
    region_b['contained_ids'] = region_b['contained_ids'] | region_a['contained_ids']
    
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
        # For transparent regions, color is None, so we need special handling
        # Only merge regions of the same color (or both transparent)
        color_c = region_c['color']
        color_b = region_b['color']
        is_c_transparent = region_c.get('is_transparent', False)
        is_b_transparent = region_b.get('is_transparent', False)
        
        if color_c != color_b:
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
        
        # Update containment list: B now contains all regions that C contained
        region_b['contained_ids'] = region_b['contained_ids'] | region_c['contained_ids']
        
        # Update color_counts for recursive merge
        _update_color_counts_for_merge(
            image_context,
            color_c,
            region_c['size'],
            is_c_transparent,
            color_b,
            is_b_transparent
        )
        
        del region_registry[c_id]
        
        same_color_neighbors_to_check.discard(c_id)


def _render_final_image(
    rgba: np.ndarray,
    region_registry: Dict[int, dict],
    region_cache: Dict[int, dict]
) -> None:
    """
    Render the final image based on the merged region graph.
    Iterates through all final merged regions and renders all original regions
    in their containment lists.
    
    Parameters
    ----------
    rgba: np.ndarray
        Image to render into (modified in place)
    region_registry: Dict[int, dict]
        Final merged region registry
    region_cache: Dict[int, dict]
        Cache of original region data (mask, color, bbox)
    """
    # Verification: ensure all original regions are accounted for
    total_contained = sum(len(region['contained_ids']) for region in region_registry.values())
    total_original = len(region_cache)
    
    if total_contained != total_original:
        error_msg = (
            f"ERROR: Verification failed! Total contained regions ({total_contained}) "
            f"does not equal total original regions ({total_original}). "
            f"This indicates a bug in containment list maintenance."
        )
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Iterate through all final merged regions
    for merged_region_id, merged_region_data in region_registry.items():
        final_color = merged_region_data['color']
        contained_ids = merged_region_data['contained_ids']
        if final_color is None and not merged_region_data.get('is_transparent', True):
            print(f"ERROR: Final color is None for merged region {merged_region_id}, skipping")
            raise RuntimeError(f"Final color is None for merged region {merged_region_id}")
        
        if contained_ids is None:
            print(f"ERROR: Contained IDs is None for merged region {merged_region_id}, skipping")
            raise RuntimeError(f"Contained IDs is None for merged region {merged_region_id}")

        # Render all original regions in the containment list
        for original_region_id in contained_ids:
            if original_region_id not in region_cache:
                print(f"ERROR: Original region {original_region_id} not found in cache, skipping")
                continue
            
            original_data = region_cache[original_region_id]
            mask = original_data['mask']
            bbox = original_data['bbox']
            x_min, y_min, width, height = bbox
            
            # Render mask pixels to final color
            if final_color is not None:
                # Non-transparent region
                rgba[y_min:y_min+height, x_min:x_min+width, 0][mask] = final_color[0]
                rgba[y_min:y_min+height, x_min:x_min+width, 1][mask] = final_color[1]
                rgba[y_min:y_min+height, x_min:x_min+width, 2][mask] = final_color[2]
                rgba[y_min:y_min+height, x_min:x_min+width, 3][mask] = 255
            else:
                # Transparent region (delete)
                rgba[y_min:y_min+height, x_min:x_min+width, 3][mask] = 0

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
        # Skip transparent regions (color is None) from color_counts
        if color is not None:
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
    
    region_registry, global_region_labels, region_cache = _build_region_graph(all_regions, result, connectivity, graph_progress_callback)
    
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
    pass_num = 0  # Track number of passes completed
    
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
            is_transparent = region_data.get('is_transparent', False)
            
            # Skip if region size is 0 (shouldn't happen, but safety check)
            if region_size == 0:
                # Mark as transparent and keep in registry for final rendering
                region_data['color'] = None
                region_data['is_transparent'] = True
                small_region_ids.discard(region_id)
                # DO NOT remove from registry - keep it for final rendering
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
                # No neighbors - mark as transparent (keep in registry for final rendering)                
                # Mark as transparent in the graph (will be rendered as transparent in final image)
                # Keep in registry so contained_ids are preserved for final rendering
                region_data['color'] = None
                region_data['is_transparent'] = True
                _update_color_counts_for_deletion(image_context, target_color, region_size, is_transparent)
                merged_count += 1
                # Remove from small_region_ids (no longer needs processing)
                small_region_ids.discard(region_id)
                # DO NOT remove from registry - keep it for final rendering
                continue
            
            # Get valid neighbors for merging (filters out transparent neighbors)
            neighbor_by_color = _get_valid_merge_neighbors(region_id, region_data, region_registry)
            
            # Check if neighbor_by_color is empty
            # This can happen if all neighbors are transparent (which we skip)
            if not neighbor_by_color:
                # No valid non-transparent neighbors to merge into
                # Delete the region (make it transparent) and update neighbors
                # Mark as transparent in the graph (will be rendered as transparent in final image)
                # Keep in registry so contained_ids are preserved for final rendering
                region_data['color'] = None
                region_data['is_transparent'] = True
                
                # Update neighbors: remove this region from their neighbor lists
                for neighbor_id in list(neighbor_ids):
                    if neighbor_id in region_registry:
                        region_registry[neighbor_id]['neighbor_ids'].discard(region_id)
                
                _update_color_counts_for_deletion(image_context, target_color, region_size, is_transparent)
                merged_count += 1
                # Remove from small_region_ids (no longer needs processing)
                small_region_ids.discard(region_id)
                # DO NOT remove from registry - keep it for final rendering
                continue
            
            # Determine which neighbor to merge to
            if len(neighbor_by_color) == 1:
                # Single neighbor color - merge to that color
                # Since all neighbors have the same color, pick the largest one for better merge efficiency
                neighbor_color = list(neighbor_by_color.keys())[0]
                neighbor_list = neighbor_by_color[neighbor_color]
                best_neighbor_id, best_neighbor = max(neighbor_list, key=lambda x: x[1]['size'])
            else:
                # Multiple neighbor colors - use scores to select the best one
                neighbor_scores = []
                for neighbor_color, neighbor_list in neighbor_by_color.items():
                    # Find largest neighbor by size (most efficient merge target)
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
                
                # Select best neighbor by score (highest first)
                best_color, best_neighbor_id, best_neighbor, _ = max(neighbor_scores, key=lambda x: x[3])
            
            # Merge region into best neighbor
            _merge_regions_in_graph(region_registry, region_id, best_neighbor_id, small_region_ids, min_size, image_context)
            merged_count += 1
        
        # Update totals for this pass
        total_merged += merged_count
        
        if progress_callback:
            progress_callback(20 + pass_num * 25, 100, f"Pass {pass_num} complete: {merged_count} regions merged")
    
    # Render final image based on merged graph
    if progress_callback:
        progress_callback(95, 100, "Rendering final image...")
    
    _render_final_image(result, region_registry, region_cache)
    
    if progress_callback:
        progress_callback(100, 100, f"Complete: {total_merged} regions merged")
    
    print(f"Region cleanup complete: {total_merged} regions merged")
    
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
    
    # Check if seed point is transparent
    is_transparent_seed = alpha[y, x] == 0
    
    if is_transparent_seed:
        # For transparent pixels, flood fill all connected transparent pixels
        # Create a mask for transparent pixels
        transparent_mask = (alpha == 0).astype(np.uint8)
        
        # Use connected components to find the transparent region
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(transparent_mask, connectivity=8)
        
        # Find which component the seed point belongs to
        seed_label = labels[y, x]
        
        if seed_label > 0:  # Found a transparent region
            # Fill all pixels in this transparent region
            region_mask = (labels == seed_label)
            result[region_mask, :3] = fill_color
            result[region_mask, 3] = 255  # Make fully opaque
    else:
        # For non-transparent pixels, use normal flood fill
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
    
    # Check if seed point is transparent
    is_transparent_seed = alpha[y, x] == 0
    
    if is_transparent_seed:
        # For transparent pixels, flood fill all connected transparent pixels
        # Simple flood fill using a stack
        stack = [(x, y)]
        visited = set()
        
        while stack:
            cx, cy = stack.pop()
            
            if (cx, cy) in visited:
                continue
                
            if (cx < 0 or cx >= rgb.shape[1] or 
                cy < 0 or cy >= rgb.shape[0]):
                continue
                
            # Only fill transparent pixels
            if alpha[cy, cx] != 0:
                continue
                
            visited.add((cx, cy))
            rgb[cy, cx] = fill_color
            alpha[cy, cx] = 255  # Make fully opaque
            
            # Add neighbors to stack
            stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    else:
        # For non-transparent pixels, use normal flood fill
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
