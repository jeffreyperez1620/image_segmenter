"""
Overlay widget for displaying and interacting with regions in the Arrange Regions step.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QColor, QBrush, QCursor, QWheelEvent
from PySide6.QtWidgets import QWidget

from model import AppState
from processing.arrange_regions import ColorRegions, RegionData
from ui.image_view import ImageView
from ui.base_overlay_view import BaseOverlayView
from ui.arrange_regions.region_pixmap_item import RegionPixmapItem
import cv2 as cv


class RegionOverlayView(BaseOverlayView):
    """Widget for handling region display and interaction."""
    
    def __init__(self, parent=None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None, region_model: Optional[Dict] = None, panel=None):
        super().__init__(image_view, app_state, parent, enable_mouse_tracking=True, use_blank_cursor=False)
        
        # Reference to panel for notifying about changes
        self._panel = panel
        
        self._region_model = region_model  # Reference to all color regions
        
        # Region items (QGraphicsItems added to scene)
        self._region_items: List[RegionPixmapItem] = []
        
        # Current color regions being displayed
        self._current_color_regions: Optional[ColorRegions] = None
        
        # Interaction state
        self._dragging_item: Optional[RegionPixmapItem] = None
        self._drag_offset: QPointF = QPointF(0, 0)
        self._hovered_item: Optional[RegionPixmapItem] = None
        self._active_item: Optional[RegionPixmapItem] = None  # Active region (shows outline and rotation handle)
        self._rotating_item: Optional[RegionPixmapItem] = None  # Currently rotating
        self._rotation_start_angle: float = 0.0
        self._rotation_handle_radius: float = 30.0  # Radius of rotation circle in pixels
        self._rotation_handle_threshold: float = 5.0  # Additional threshold for easier clicking
        
        # Image bounds for constraint checking
        self._image_bounds: Optional[QRectF] = None
    
    def set_regions(self, color_regions: Optional[ColorRegions]) -> None:
        """Set the regions to display."""
        # Clear existing items
        self._clear_region_items()
        
        self._current_color_regions = color_regions
        
        if color_regions is None:
            return
        
        # Update image bounds
        self._update_image_bounds()
        
        # Create region items and add to scene
        if self._image_view and hasattr(self._image_view, '_scene'):
            for i, region_data in enumerate(color_regions.regions):
                item = RegionPixmapItem(region_data)
                # Set Z-value based on index (later regions on top)
                item.setZValue(100 + i)
                self._image_view._scene.addItem(item)
                self._region_items.append(item)
    
    def _clear_region_items(self) -> None:
        """Clear all region items from the scene."""
        if self._image_view and hasattr(self._image_view, '_scene'):
            for item in self._region_items:
                self._image_view._scene.removeItem(item)
        self._region_items.clear()
        self._hovered_item = None
        self._dragging_item = None
        self._active_item = None
        self._rotating_item = None
    
    def _update_image_bounds(self) -> None:
        """Update image bounds from base image."""
        if self._app_state and self._app_state.base_image is not None:
            h, w = self._app_state.base_image.shape[:2]
            self._image_bounds = QRectF(0, 0, w, h)
        else:
            self._image_bounds = None
    
    def _widget_to_scene(self, widget_pos) -> QPointF:
        """Convert widget coordinates to scene coordinates."""
        if self._image_view:
            return self._image_view._view.mapToScene(widget_pos)
        return QPointF(widget_pos)
    
    def _scene_to_widget(self, scene_pos: QPointF) -> QPointF:
        """Convert scene coordinates to widget coordinates."""
        if self._image_view:
            return self._image_view._view.mapFromScene(scene_pos)
        return scene_pos
    
    def _find_item_at_position(self, scene_pos: QPointF) -> Optional[RegionPixmapItem]:
        """Find the topmost region item at the given scene position.
        
        This method accounts for rotation and other transforms by using Qt's
        coordinate transformation methods.
        """
        # Check items in reverse order (top to bottom)
        for item in reversed(self._region_items):
            # Use Qt's transform to convert scene position to item's local coordinates
            # This automatically accounts for rotation, scaling, and other transforms
            local_pos = item.mapFromScene(scene_pos)
            
            # Check if point is within pixmap bounds (in local coordinates)
            pixmap = item.pixmap()
            if pixmap.isNull():
                continue
                
            x = int(local_pos.x())
            y = int(local_pos.y())
            
            if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                # Check if point hits a non-transparent pixel
                image = pixmap.toImage()
                pixel_color = image.pixelColor(x, y)
                if pixel_color.alpha() > 0:
                    return item
        
        return None
    
    def leaveEvent(self, event) -> None:
        """Handle leave event to clear hover state."""
        super().leaveEvent(event)
        # Clear hover state when mouse leaves
        if self._hovered_item:
            self._hovered_item.set_hovered(False)
            self._hovered_item = None
        # Reset cursor when mouse leaves
        if self._image_view and hasattr(self._image_view, '_view'):
            self._image_view._view.viewport().unsetCursor()
    
    
    def _delta_to_centroid(self, scene_pos: QPointF, item: RegionPixmapItem) -> QPointF:
        """Get the (dx, dy) vector from the region centroid to the given scene position."""
        centroid_item = item.get_centroid()
        centroid_scene = item.mapToScene(centroid_item)
        delta = scene_pos - centroid_scene
        return delta
    
    def _is_on_rotation_handle(self, scene_pos: QPointF, item: Optional[RegionPixmapItem]) -> bool:
        """Check if the point is on the rotation handle circle.
        
        We work in widget coordinates for hit testing to match the visual representation.
        The radius is defined in scene coordinates (30 pixels), but we convert everything
        to widget coordinates to properly account for zoom.
        """
        if item is None or not item.is_active():
            return False
        
        # Convert scene position to widget coordinates
        scene_pos_widget = self._scene_to_widget(scene_pos)
        
        # Get the centroid in widget coordinates
        centroid_item = item.get_centroid()
        centroid_scene = item.mapToScene(centroid_item)
        centroid_widget = self._scene_to_widget(centroid_scene)
        
        # Calculate distance in widget coordinates
        delta_widget = scene_pos_widget - centroid_widget
        distance_widget = np.sqrt(delta_widget.x() ** 2 + delta_widget.y() ** 2)
        
        # Get the current zoom/scale factor from the view
        if self._image_view and hasattr(self._image_view, '_view'):
            transform = self._image_view._view.transform()
            scale_factor = abs(transform.m11())
            if scale_factor < 0.001:
                scale_factor = 1.0
        else:
            scale_factor = 1.0
        
        # Scale the rotation handle radius and threshold to widget coordinates
        # The radius is in scene pixels, so multiply by scale factor to get widget pixels
        scaled_radius = self._rotation_handle_radius * scale_factor
        scaled_threshold = self._rotation_handle_threshold * scale_factor
        
        # Check if distance (in widget coordinates) is within the scaled radius range
        min_distance = max(0, scaled_radius - scaled_threshold)
        max_distance = scaled_radius + scaled_threshold
        is_on_handle = min_distance <= distance_widget <= max_distance
        
        return is_on_handle
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press event."""
        if event.button() != Qt.LeftButton:
            # Forward non-left button events to ImageView for right-click panning
            self._forward_event_to_image_view(event)
            return
        
        scene_pos = self._widget_to_scene(event.position().toPoint())
        item = self._find_item_at_position(scene_pos)
        
        # Clicked on empty space or transparent pixel, check if on rotation handle
        if item is None and self._active_item:
            if self._is_on_rotation_handle(scene_pos, self._active_item):
                # Start rotation
                delta = self._delta_to_centroid(scene_pos, self._active_item)
                self._rotating_item = self._active_item
                self._rotation_start_angle = np.arctan2(delta.y(), delta.x()) * 180.0 / np.pi
                event.accept()
                return
            else:
                # Clicked on transparent pixel, clear active region
                if self._active_item:
                    self._active_item.set_active(False)
                self._active_item = None
                self.update()
                return
        
        if item:
            # Set as active region if not already active.
            if self._active_item != item:
                if self._active_item:
                    self._active_item.set_active(False)
                item.set_active(True)
                self._active_item = item
                # If newly active, start dragging
                self._dragging_item = item
                item_pos = item.pos()
                self._drag_offset = scene_pos - item_pos
                event.accept()
                return
            # Check if clicking on rotation handle
            if self._is_on_rotation_handle(scene_pos, item):
                # Start rotation
                delta = self._delta_to_centroid(scene_pos, item)
                self._rotating_item = item
                self._rotation_start_angle = np.arctan2(delta.y(), delta.x()) * 180.0 / np.pi
                event.accept()
            else:
                # Start dragging
                self._dragging_item = item
                item_pos = item.pos()
                self._drag_offset = scene_pos - item_pos
                event.accept()
        else:
            # Clicked on empty space or transparent pixel - clear active region
            if self._active_item:
                self._active_item.set_active(False)
                self._active_item = None
            # Forward event to ImageView for panning
            self._forward_event_to_image_view(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move event."""
        # Forward right button drag events for panning
        if event.buttons() & Qt.RightButton:
            self._forward_event_to_image_view(event)
            return
        
        if not self._region_items:
            # No regions to interact with, forward event for panning
            self._forward_event_to_image_view(event)
            return
        
        scene_pos = self._widget_to_scene(event.position().toPoint())
        
        if self._rotating_item and event.buttons() & Qt.LeftButton:
            # Handle rotation
            delta = self._delta_to_centroid(scene_pos, self._rotating_item)
            current_angle = np.arctan2(delta.y(), delta.x()) * 180.0 / np.pi
            # Calculate rotation delta
            rotation_delta = current_angle - self._rotation_start_angle
            
            # Normalize to -180 to 180 range
            while rotation_delta > 180:
                rotation_delta -= 360
            while rotation_delta < -180:
                rotation_delta += 360
            
            # Update rotation
            new_rotation = self._rotating_item._region_data.rotation + rotation_delta
            # Normalize to 0-360 range
            new_rotation = new_rotation % 360.0
            if new_rotation < 0:
                new_rotation += 360.0
            
            self._rotating_item._region_data.rotation = new_rotation
            self._rotating_item.update_rotation()
            
            # Update start angle for next calculation
            self._rotation_start_angle = current_angle
            
            # Notify panel that changes have been made
            if self._panel:
                self._panel._on_region_changed()
            
            # Don't update working image during rotation - only update on release
            # This prevents duplicate rendering (graphics item + working image)
            # The graphics item provides smooth visual feedback during rotation
            
            event.accept()
        elif self._dragging_item and event.buttons() & Qt.LeftButton:
            # Update position
            new_pos = scene_pos - self._drag_offset
            
            # Constrain to image bounds
            if self._image_bounds is not None:
                item_rect = QRectF(0, 0, self._dragging_item.pixmap().width(), self._dragging_item.pixmap().height())
                min_x = 0
                min_y = 0
                max_x = self._image_bounds.width() - item_rect.width()
                max_y = self._image_bounds.height() - item_rect.height()
                
                new_pos.setX(max(min_x, min(max_x, new_pos.x())))
                new_pos.setY(max(min_y, min(max_y, new_pos.y())))
            
            # Update item position
            self._dragging_item.setPos(new_pos)
            self._dragging_item.update_position()
            # Notify panel that changes have been made
            if self._panel:
                self._panel._on_region_changed()
            
            event.accept()
        else:
            # Not dragging/rotating - check for hover or forward for panning
            item = self._find_item_at_position(scene_pos)
            
            if item:
                # Update hover state (but don't override active item outline)
                if item != self._hovered_item:
                    if self._hovered_item:
                        self._hovered_item.set_hovered(False)
                    item.set_hovered(True)
                    self._hovered_item = item
                event.accept()
            else:
                # Not over a region - forward for panning
                self._forward_event_to_image_view(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release event."""
        if event.button() == Qt.LeftButton:
            if self._dragging_item:
                self._dragging_item = None
                self._drag_offset = QPointF(0, 0)
            if self._rotating_item:
                self._rotating_item = None
                self._rotation_start_angle = 0.0
            event.accept()
        else:
            # Forward non-left button events to ImageView
            self._forward_event_to_image_view(event)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle wheel events for zooming."""
        # Forward all wheel events to the ImageView for zoom functionality
        self._forward_wheel_event_to_image_view(event)
    
    def paintEvent(self, event) -> None:
        """Paint the rotation handles for active regions."""
        if not self._active or not self._active_item or not self._active_item.is_active():
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get active item's centroid in scene coordinates
        centroid_item = self._active_item.get_centroid()
        centroid_scene = self._active_item.mapToScene(centroid_item)
        centroid_widget = self._scene_to_widget(centroid_scene)
        
        # Get the current zoom/scale factor from the view
        if self._image_view and hasattr(self._image_view, '_view'):
            # Get the transform matrix and extract the scale factor
            transform = self._image_view._view.transform()
            # The scale factor is the square root of the determinant of the 2x2 matrix
            # For uniform scaling, we can use m11 or m22 (they should be equal)
            scale_factor = transform.m11()
        else:
            scale_factor = 1.0
        
        # Scale the rotation handle radius and other dimensions by the zoom factor
        scaled_radius = self._rotation_handle_radius * scale_factor
        scaled_indicator_length = (self._rotation_handle_radius - 5) * scale_factor
        scaled_arrow_size = 8 * scale_factor
        scaled_pen_width = max(1, int(2 * scale_factor))  # Minimum 1 pixel
        scaled_line_width = max(1, int(3 * scale_factor))  # Minimum 1 pixel
        
        # Draw rotation circle
        pen = QPen(QColor(255, 255, 0), scaled_pen_width)  # Yellow circle
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centroid_widget, scaled_radius, scaled_radius)
        
        # Draw rotation indicator (arrow/line pointing in rotation direction)
        rotation_rad = np.radians(self._active_item._region_data.rotation)
        end_x = centroid_widget.x() + np.cos(rotation_rad) * scaled_indicator_length
        end_y = centroid_widget.y() + np.sin(rotation_rad) * scaled_indicator_length
        
        # Draw line from center to rotation angle
        pen = QPen(QColor(255, 255, 0), scaled_line_width)  # Thicker yellow line
        painter.setPen(pen)
        painter.drawLine(centroid_widget, QPointF(end_x, end_y))
        
        # Draw arrowhead at the end
        arrow_angle = np.pi / 6  # 30 degrees
        arrow1_x = end_x - scaled_arrow_size * np.cos(rotation_rad - arrow_angle)
        arrow1_y = end_y - scaled_arrow_size * np.sin(rotation_rad - arrow_angle)
        arrow2_x = end_x - scaled_arrow_size * np.cos(rotation_rad + arrow_angle)
        arrow2_y = end_y - scaled_arrow_size * np.sin(rotation_rad + arrow_angle)
        
        painter.drawLine(QPointF(end_x, end_y), QPointF(arrow1_x, arrow1_y))
        painter.drawLine(QPointF(end_x, end_y), QPointF(arrow2_x, arrow2_y))
    
