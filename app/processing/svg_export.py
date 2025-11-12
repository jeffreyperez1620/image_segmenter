"""
Functions for exporting arranged regions to SVG format.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy.interpolate import splprep, splev
from processing.arrange_regions import RegionData


def render_regions_to_image(regions: List[RegionData]) -> Optional[np.ndarray]:
    """
    Render a list of regions (with their positions and rotations) to a single image.
    The image size will be the bounding box of all regions.
    
    Parameters
    ----------
    regions: List[RegionData]
        List of regions to render
        
    Returns
    -------
    np.ndarray or None
        RGBA image containing all rendered regions, or None if no regions provided
    """
    if not regions:
        return None
    
    # Calculate bounding box of all regions (accounting for rotation)
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for region in regions:
        region_img = region.image
        h, w = region_img.shape[:2]
        x, y = region.position
        rotation = region.rotation
        
        # Calculate corners of the rotated region
        # Center of the region image
        center_x = w / 2.0
        center_y = h / 2.0
        
        # Corners relative to center
        corners = np.array([
            [-center_x, -center_y],  # Top-left
            [w - center_x, -center_y],  # Top-right
            [w - center_x, h - center_y],  # Bottom-right
            [-center_x, h - center_y]  # Bottom-left
        ])
        
        # Rotate corners
        if rotation != 0:
            angle_rad = np.radians(rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            corners = corners @ rotation_matrix.T
        
        # Translate to world coordinates
        corners[:, 0] += x + center_x
        corners[:, 1] += y + center_y
        
        # Update bounding box
        min_x = min(min_x, np.min(corners[:, 0]))
        min_y = min(min_y, np.min(corners[:, 1]))
        max_x = max(max_x, np.max(corners[:, 0]))
        max_y = max(max_y, np.max(corners[:, 1]))
    
    # Add some padding
    padding = 10
    min_x = int(np.floor(min_x)) - padding
    min_y = int(np.floor(min_y)) - padding
    max_x = int(np.ceil(max_x)) + padding
    max_y = int(np.ceil(max_y)) + padding
    
    # Calculate canvas size
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    
    if canvas_width <= 0 or canvas_height <= 0:
        return None
    
    # Create canvas
    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    
    # Render each region
    for region in regions:
        region_img = region.image
        h, w = region_img.shape[:2]
        x, y = region.position  # Top-left position
        rotation = region.rotation
        
        # Calculate the center of the region in world coordinates
        center_x_world = x + w / 2.0
        center_y_world = y + h / 2.0
        
        # Rotate the region image around its center
        if abs(rotation) > 0.01:  # Use small epsilon instead of != 0
            # Get rotation matrix (center is relative to image)
            # Note: Qt uses clockwise rotation for positive angles, but OpenCV uses counterclockwise
            # So we need to negate the angle to match Qt's rotation direction
            center = (w / 2.0, h / 2.0)
            rotation_matrix = cv.getRotationMatrix2D(center, -rotation, 1.0)
            
            # Calculate bounding box of rotated image by transforming corners
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ], dtype=np.float32)
            
            # Transform corners using the rotation matrix
            ones = np.ones(shape=(len(corners), 1))
            corners_ones = np.hstack([corners, ones])
            transformed_corners = rotation_matrix @ corners_ones.T
            transformed_corners = transformed_corners.T
            
            # Calculate bounding box of transformed corners
            min_x_rot = np.min(transformed_corners[:, 0])
            max_x_rot = np.max(transformed_corners[:, 0])
            min_y_rot = np.min(transformed_corners[:, 1])
            max_y_rot = np.max(transformed_corners[:, 1])
            
            # Calculate size needed for rotated image
            rot_w = int(np.ceil(max_x_rot - min_x_rot))
            rot_h = int(np.ceil(max_y_rot - min_y_rot))
            
            # Calculate where the original center ends up after rotation (before translation adjustment)
            center_point = np.array([[center[0]], [center[1]], [1]], dtype=np.float32)
            center_rotated = rotation_matrix @ center_point
            center_rotated_x = center_rotated[0, 0]
            center_rotated_y = center_rotated[1, 0]
            
            # Adjust translation to shift the bounding box so min corner is at (0,0)
            # The rotation matrix rotates around the center, but we need to shift
            # the result so the entire rotated image fits in a positive coordinate space
            tx = -min_x_rot
            ty = -min_y_rot
            rotation_matrix[0, 2] += tx
            rotation_matrix[1, 2] += ty
            
            # Calculate where the center is now after translation adjustment
            center_after_translation_x = center_rotated_x + tx
            center_after_translation_y = center_rotated_y + ty
            
            # Rotate the region image
            rotated_img = cv.warpAffine(
                region_img,
                rotation_matrix,
                (rot_w, rot_h),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_TRANSPARENT
            )
            
            # Calculate where to place the rotated image
            # The center of the rotated image is at center_after_translation_x, center_after_translation_y
            # in the rotated_img coordinate system. We want this center to be at center_x_world, center_y_world
            # in world coordinates (relative to the canvas, which starts at min_x, min_y)
            place_x = int(center_x_world - center_after_translation_x) - min_x
            place_y = int(center_y_world - center_after_translation_y) - min_y
        else:
            rotated_img = region_img
            place_x = int(x - min_x)
            place_y = int(y - min_y)
            rot_w = w
            rot_h = h
        
        # Place the rotated region on the canvas with alpha blending
        region_y_start = max(0, place_y)
        region_y_end = min(canvas_height, place_y + rot_h)
        region_x_start = max(0, place_x)
        region_x_end = min(canvas_width, place_x + rot_w)
        
        canvas_y_start = region_y_start
        canvas_y_end = region_y_end
        canvas_x_start = region_x_start
        canvas_x_end = region_x_end
        
        region_y_offset = region_y_start - place_y
        region_x_offset = region_x_start - place_x
        
        if (region_y_end > region_y_start and region_x_end > region_x_start and
            region_y_offset >= 0 and region_x_offset >= 0 and
            region_y_offset + (canvas_y_end - canvas_y_start) <= rot_h and
            region_x_offset + (canvas_x_end - canvas_x_start) <= rot_w):
            
            region_patch = rotated_img[
                region_y_offset:region_y_offset + (canvas_y_end - canvas_y_start),
                region_x_offset:region_x_offset + (canvas_x_end - canvas_x_start)
            ]
            
            canvas_patch = canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end]
            
            # Alpha blend - filter out semi-transparent edge pixels from rotation interpolation
            alpha = region_patch[:, :, 3:4] / 255.0
            # Only blend pixels with alpha >= threshold to avoid interpolation artifacts
            valid_mask = region_patch[:, :, 3] >= 128
            
            # For valid pixels, use alpha blending
            canvas_patch[valid_mask, :3] = (canvas_patch[valid_mask, :3] * (1 - alpha[valid_mask, 0:1]) + 
                                           region_patch[valid_mask, :3] * alpha[valid_mask, 0:1]).astype(np.uint8)
            canvas_patch[valid_mask, 3] = np.maximum(canvas_patch[valid_mask, 3], region_patch[valid_mask, 3])
    
    return canvas


def calculate_regions_bounding_box(regions: List[RegionData]) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate the bounding box of a list of regions (accounting for rotation).
    
    Parameters
    ----------
    regions: List[RegionData]
        List of regions
        
    Returns
    -------
    Tuple[float, float, float, float] or None
        (min_x, min_y, width, height) of the bounding box, or None if no regions
    """
    if not regions:
        return None
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for region in regions:
        region_img = region.image
        h, w = region_img.shape[:2]
        x, y = region.position
        rotation = region.rotation
        
        # Calculate corners of the rotated region
        center_x = w / 2.0
        center_y = h / 2.0
        
        corners = np.array([
            [-center_x, -center_y],
            [w - center_x, -center_y],
            [w - center_x, h - center_y],
            [-center_x, h - center_y]
        ])
        
        # Rotate corners
        # Note: Qt uses clockwise rotation for positive angles, but standard rotation matrices
        # rotate counterclockwise, so we need to negate the angle
        if abs(rotation) > 0.01:
            angle_rad = np.radians(-rotation)  # Negate to match Qt's clockwise rotation
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            corners = corners @ rotation_matrix.T
        
        # Translate to world coordinates
        corners[:, 0] += x + center_x
        corners[:, 1] += y + center_y
        
        # Update bounding box
        min_x = min(min_x, np.min(corners[:, 0]))
        min_y = min(min_y, np.min(corners[:, 1]))
        max_x = max(max_x, np.max(corners[:, 0]))
        max_y = max(max_y, np.max(corners[:, 1]))
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 0 or height <= 0:
        return None
    
    return (min_x, min_y, width, height)


def export_regions_to_svg(
    regions: List[RegionData],
    output_width: float,
    output_height: float,
    margin_width: float,
    margin_height: float,
    units: str,
    output_path: str,
    enable_smoothing: bool = False
) -> bool:
    """
    Export regions to SVG format as outlined contours.
    
    Parameters
    ----------
    regions: List[RegionData]
        List of regions with their positions and rotations
    output_width: float
        Desired output width in the specified units
    output_height: float
        Desired output height in the specified units
    margin_width: float
        Margin width in the specified units
    margin_height: float
        Margin height in the specified units
    units: str
        Units of measurement ("in", "cm", or "mm")
    output_path: str
        Path to save the SVG file
    enable_smoothing: bool
        If True, apply contour smoothing using cv2.approxPolyDP()
        
    Returns
    -------
    bool
        True if export was successful, False otherwise
    """
    if not regions:
        return False
    
    # Calculate the bounding box of all regions to determine the source image size
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for region in regions:
        region_img = region.image
        h, w = region_img.shape[:2]
        x, y = region.position
        rotation = region.rotation
        
        # Calculate corners of the rotated region
        center_x = w / 2.0
        center_y = h / 2.0
        
        corners = np.array([
            [-center_x, -center_y],
            [w - center_x, -center_y],
            [w - center_x, h - center_y],
            [-center_x, h - center_y]
        ])
        
        # Rotate corners
        # Note: Qt uses clockwise rotation for positive angles, but standard rotation matrices
        # rotate counterclockwise, so we need to negate the angle
        if abs(rotation) > 0.01:
            angle_rad = np.radians(-rotation)  # Negate to match Qt's clockwise rotation
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            corners = corners @ rotation_matrix.T
        
        # Translate to world coordinates
        corners[:, 0] += x + center_x
        corners[:, 1] += y + center_y
        
        # Update bounding box
        min_x = min(min_x, np.min(corners[:, 0]))
        min_y = min(min_y, np.min(corners[:, 1]))
        max_x = max(max_x, np.max(corners[:, 0]))
        max_y = max(max_y, np.max(corners[:, 1]))
    
    # Calculate source dimensions from bounding box (no padding - margins are handled separately)
    min_x = int(np.floor(min_x))
    min_y = int(np.floor(min_y))
    max_x = int(np.ceil(max_x))
    max_y = int(np.ceil(max_y))
    
    source_width = max_x - min_x
    source_height = max_y - min_y
    
    if source_width <= 0 or source_height <= 0:
        return False
    
    # Calculate SVG dimensions (output size + margins)
    svg_width = output_width + 2 * margin_width
    svg_height = output_height + 2 * margin_height
    
    # Calculate scale factors
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    scale = min(scale_x, scale_y)  # Maintain aspect ratio
    
    # Calculate scaled dimensions and centering offset
    scaled_width = source_width * scale
    scaled_height = source_height * scale
    offset_x = margin_width + (output_width - scaled_width) / 2.0
    offset_y = margin_height + (output_height - scaled_height) / 2.0
    
    # Create SVG root element
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', f'{svg_width}{units}')
    svg.set('height', f'{svg_height}{units}')
    svg.set('viewBox', f'0 0 {svg_width} {svg_height}')
    
    # Process each region
    for region in regions:
        region_img = region.image
        h, w = region_img.shape[:2]
        x, y = region.position
        rotation = region.rotation
        
        # Extract alpha channel as mask
        if region_img.shape[2] == 4:
            alpha = region_img[:, :, 3]
        else:
            alpha = np.ones((h, w), dtype=np.uint8) * 255
        
        # Create binary mask
        mask = (alpha > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Process each contour
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Apply smoothing if enabled
            if enable_smoothing:
                # First, simplify the contour to reduce points
                perimeter = cv.arcLength(contour, True)
                epsilon = 0.01 * perimeter
                contour = cv.approxPolyDP(contour, epsilon, True)
                if len(contour) < 3:
                    continue
                
                # Reshape contour for spline interpolation
                contour_reshaped = contour.reshape(-1, 2)
                
                # Extract x and y coordinates (use different variable names to avoid collision)
                contour_x = contour_reshaped[:, 0].astype(np.float64)
                contour_y = contour_reshaped[:, 1].astype(np.float64)
                
                # Close the contour by adding the first point at the end if not already closed
                if not np.array_equal(contour_reshaped[0], contour_reshaped[-1]):
                    contour_x = np.append(contour_x, contour_x[0])
                    contour_y = np.append(contour_y, contour_y[0])
                
                # Fit B-spline to the contour
                # s parameter controls smoothness: smaller = closer to original, larger = smoother
                # per=1 means periodic (closed contour)
                try:
                    tck, u = splprep([contour_x, contour_y], s=len(contour) * 0.5, per=1)
                    
                    # Generate smoothed points
                    # Use fewer points than original for smoother result
                    num_points = max(10, min(len(contour), 50))  # Between 10 and 50 points
                    u_new = np.linspace(0, 1, num_points)
                    x_new, y_new = splev(u_new, tck, der=0)
                    
                    # Convert back to contour format
                    contour = np.array([[int(round(xi)), int(round(yi))] for xi, yi in zip(x_new, y_new)], dtype=np.int32).reshape(-1, 1, 2)
                except Exception:
                    # If spline fitting fails, use the simplified contour
                    pass
            
            # Convert contour points to numpy array (shape: N, 1, 2)
            points = contour.reshape(-1, 2).astype(np.float32)
            
            # Transform points: apply rotation, then translate to world coordinates
            # The region position (x, y) is the top-left corner of the unrotated region
            # We need to rotate around the region's center, then translate to world coordinates
            center_x = w / 2.0
            center_y = h / 2.0
            region_center_world_x = x + center_x
            region_center_world_y = y + center_y
            
            # Translate points to center origin (relative to region center)
            points_centered = points - np.array([center_x, center_y])
            
            # Apply rotation if needed
            # Note: Qt rotates clockwise for positive angles. Since we're working with
            # image coordinates (y increases downward), we use the rotation angle directly.
            # The standard rotation matrix will produce the correct clockwise rotation
            # when applied in y-down coordinate space.
            if abs(rotation) > 0.01:
                angle_rad = np.radians(rotation)  # Use rotation directly
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                # Standard rotation matrix (counterclockwise in math coords, clockwise in y-down coords)
                rotation_matrix = np.array([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])
                # Apply rotation: points_centered @ rotation_matrix.T
                # This rotates points around the origin
                points_centered = points_centered @ rotation_matrix.T
            
            # Translate to world coordinates (add back the region center)
            points_world = points_centered + np.array([region_center_world_x, region_center_world_y])
            
            # Transform to SVG coordinates: subtract min bounds, scale, add offset
            points_svg = (points_world - np.array([min_x, min_y])) * scale + np.array([offset_x, offset_y])
            
            # Create SVG path
            path = ET.SubElement(svg, 'path')
            path_data = f'M {points_svg[0, 0]:.3f},{points_svg[0, 1]:.3f}'
            for i in range(1, len(points_svg)):
                path_data += f' L {points_svg[i, 0]:.3f},{points_svg[i, 1]:.3f}'
            path_data += ' Z'  # Close path
            
            path.set('d', path_data)
            path.set('fill', 'none')
            path.set('stroke', 'black')
            # Use a very small stroke width for fine engraving lines
            # Make it very small and scale-independent for consistent appearance
            # 0.01 units should be thin enough for most engraving applications
            stroke_width = 0.01
            path.set('stroke-width', str(stroke_width))
    
    # Write SVG to file
    try:
        # Pretty print the XML
        xml_str = ET.tostring(svg, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')
        
        # Remove the XML declaration line (SVG doesn't need it)
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        pretty_xml = '\n'.join(lines).strip()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        return True
    except Exception as e:
        print(f"Error writing SVG file: {e}")
        return False

