"""
Overlay widget for displaying and interacting with regions in the Arrange Regions step.
"""

from __future__ import annotations

from typing import Optional, List
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QMouseEvent, QPainter
from PySide6.QtWidgets import QWidget

from model import AppState
from processing.arrange_regions import ColorRegions, RegionData
from ui.image_view import ImageView
from ui.arrange_regions.region_pixmap_item import RegionPixmapItem


class RegionOverlayView(QWidget):
    """Widget for handling region display and interaction."""
    
    def __init__(self, parent=None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None):
        super().__init__(parent)
        
        # Make this widget transparent and able to receive mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")
        
        self._image_view = image_view
        self._app_state = app_state
        
        # Region items (QGraphicsItems added to scene)
        self._region_items: List[RegionPixmapItem] = []
        
        # Current color regions being displayed
        self._current_color_regions: Optional[ColorRegions] = None
        
        # Interaction state
        self._dragging_item: Optional[RegionPixmapItem] = None
        self._drag_offset: QPointF = QPointF(0, 0)
        self._hovered_item: Optional[RegionPixmapItem] = None
        
        # Image bounds for constraint checking
        self._image_bounds: Optional[QRectF] = None
        
        # Position the overlay view over the image view
        if self._image_view:
            self.setGeometry(0, 0, self._image_view.width(), self._image_view.height())
            self.raise_()
    
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
    
    def _find_item_at_position(self, scene_pos: QPointF) -> Optional[RegionPixmapItem]:
        """Find the topmost region item at the given scene position."""
        # Check items in reverse order (top to bottom)
        for item in reversed(self._region_items):
            item_pos = item.pos()
            item_rect = QRectF(item_pos.x(), item_pos.y(), item.pixmap().width(), item.pixmap().height())
            
            # Check if point is within bounding rect
            if item_rect.contains(scene_pos):
                # Check if point hits a non-transparent pixel
                local_pos = scene_pos - item_pos
                x = int(local_pos.x())
                y = int(local_pos.y())
                
                if 0 <= x < item.pixmap().width() and 0 <= y < item.pixmap().height():
                    image = item.pixmap().toImage()
                    pixel_color = image.pixelColor(x, y)
                    if pixel_color.alpha() > 0:
                        return item
        
        return None
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press event."""
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        
        scene_pos = self._widget_to_scene(event.position().toPoint())
        item = self._find_item_at_position(scene_pos)
        
        if item:
            self._dragging_item = item
            item_pos = item.pos()
            self._drag_offset = scene_pos - item_pos
            event.accept()
        else:
            event.ignore()
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move event."""
        scene_pos = self._widget_to_scene(event.position().toPoint())
        
        if self._dragging_item and event.buttons() & Qt.LeftButton:
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
            event.accept()
        else:
            # Update hover state
            item = self._find_item_at_position(scene_pos)
            
            if item != self._hovered_item:
                if self._hovered_item:
                    self._hovered_item.set_hovered(False)
                if item:
                    item.set_hovered(True)
                self._hovered_item = item
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release event."""
        if event.button() == Qt.LeftButton and self._dragging_item:
            self._dragging_item = None
            self._drag_offset = QPointF(0, 0)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

