from __future__ import annotations

from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QColor, QBrush, QCursor
from PySide6.QtWidgets import QWidget

from model import AppState
from ui.image_view import ImageView
from ui.base_overlay_view import BaseOverlayView


class FloodFillView(BaseOverlayView):
    """A transparent overlay widget for flood fill operations."""
    
    flood_fill_requested = Signal(QPoint, QColor)
    
    def __init__(self, image_view: ImageView, app_state: AppState, parent: Optional[QWidget] = None):
        super().__init__(image_view, app_state, parent, enable_mouse_tracking=True, use_blank_cursor=True)
        self._fill_color = None
        self._mouse_pos = QPoint(0, 0)
        self._show_cursor = False
    
    def set_active(self, active: bool) -> None:
        """Activate or deactivate the flood fill view."""
        super().set_active(active)
        if not active:
            self._fill_color = None
            self.update()
    
    def set_fill_color(self, color: QColor) -> None:
        """Set the color to use for flood filling."""
        self._fill_color = color
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events for flood fill."""
        if not self._active:
            self._forward_event_to_image_view(event)
            return
        
        if event.button() == Qt.LeftButton and self._fill_color is not None:
            # Get the image coordinates for the flood fill
            scene_pos = self._image_view._view.mapToScene(event.pos())
            pixmap_item = self._image_view._pix_item
            if pixmap_item is not None:
                item_pos = pixmap_item.pos()
                item_size = pixmap_item.pixmap().size()
                
                # Convert to image coordinates
                x = int((scene_pos.x() - item_pos.x()) * item_size.width() / pixmap_item.pixmap().width())
                y = int((scene_pos.y() - item_pos.y()) * item_size.height() / pixmap_item.pixmap().height())
                
                # Emit the flood fill request with image coordinates
                self.flood_fill_requested.emit(QPoint(x, y), self._fill_color)
        else:
            self._forward_event_to_image_view(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        self._mouse_pos = event.pos()
        self._forward_event_to_image_view(event)
        
        # Always show cursor when active (within widget bounds)
        # The bounds check is only used to determine if operations are allowed
        if self._active:
            self._show_cursor = True
        else:
            self._show_cursor = False
        
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        self._forward_event_to_image_view(event)
    
    def wheelEvent(self, event) -> None:
        """Handle wheel events for zooming."""
        self._forward_wheel_event_to_image_view(event)
    
    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        self._forward_key_event_to_image_view(event)
    
    def paintEvent(self, event) -> None:
        """Paint the flood fill preview."""
        if not self._active or self._fill_color is None:
            return
        
        # Only show cursor preview if mouse is within widget bounds
        if not self._show_cursor:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a crosshair at the mouse position
        crosshair_size = 12
        painter.setPen(QPen(Qt.white, 2))
        painter.drawLine(self._mouse_pos.x() - crosshair_size, self._mouse_pos.y(), 
                        self._mouse_pos.x() + crosshair_size, self._mouse_pos.y())
        painter.drawLine(self._mouse_pos.x(), self._mouse_pos.y() - crosshair_size,
                        self._mouse_pos.x(), self._mouse_pos.y() + crosshair_size)
        
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(self._mouse_pos.x() - crosshair_size, self._mouse_pos.y(), 
                        self._mouse_pos.x() + crosshair_size, self._mouse_pos.y())
        painter.drawLine(self._mouse_pos.x(), self._mouse_pos.y() - crosshair_size,
                        self._mouse_pos.x(), self._mouse_pos.y() + crosshair_size)
        
        # Draw a small preview of the fill color near the cursor
        if self._fill_color is not None:
            radius = 8
            offset_x = 5
            offset_y = 5
            
            preview_x = min(self._mouse_pos.x() + offset_x, self.width() - radius * 2 - 5)
            preview_y = min(self._mouse_pos.y() + offset_y, self.height() - radius * 2 - 5)
            
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(self._fill_color))
            painter.drawEllipse(preview_x, preview_y, radius * 2, radius * 2)
            
            # Draw a small border
            painter.setPen(QPen(Qt.white, 1))
            painter.drawEllipse(preview_x + 1, preview_y + 1, (radius - 1) * 2, (radius - 1) * 2)
    
    def _is_mouse_within_image_bounds(self, pos: QPoint) -> bool:
        """Check if the mouse position is within the image bounds."""
        # Convert widget coordinates to scene coordinates
        scene_pos = self._image_view._view.mapToScene(pos)
        
        # Get the image from app state
        display_image = self._image_view.get_current_display_image()
        if display_image is None:
            return False
        
        # Convert scene coordinates to image coordinates
        pixmap_item = self._image_view._pix_item
        if pixmap_item is None:
            return False
        
        # Get the pixmap item's position and size
        item_pos = pixmap_item.pos()
        item_size = pixmap_item.pixmap().size()
        
        # Calculate image coordinates
        x = int((scene_pos.x() - item_pos.x()) * item_size.width() / pixmap_item.pixmap().width())
        y = int((scene_pos.y() - item_pos.y()) * item_size.height() / pixmap_item.pixmap().height())
        
        # Check bounds
        return 0 <= x < display_image.shape[1] and 0 <= y < display_image.shape[0]
    
    def showEvent(self, event) -> None:
        """Handle show events - update geometry when shown."""
        super().showEvent(event)
        self._update_geometry()
