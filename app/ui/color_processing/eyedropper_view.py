from __future__ import annotations

from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QMouseEvent
from PySide6.QtWidgets import QWidget

from model import AppState
from ui.image_view import ImageView


class EyedropperView(QWidget):
    """A transparent overlay widget that handles eyedropper functionality for color picking."""
    
    # Signals
    color_picked = Signal(QColor)
    eyedropper_cancelled = Signal()
    
    def __init__(self, image_view: ImageView, app_state: AppState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_view = image_view
        self._app_state = app_state
        self._active = False
        self._preview_color = None
        self._mouse_pos = QPoint(0, 0)
        
        # Make the widget transparent and capture mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setMouseTracking(True)
        # Hide the system cursor when this widget is active
        self.setCursor(Qt.BlankCursor)
        
        # Don't set geometry here - it will be set when activated
        # The ImageView might not have its final size yet during initialization
        
        # Initially hidden
        self.hide()
    
    def set_active(self, active: bool) -> None:
        """Activate or deactivate the eyedropper."""
        self._active = active
        if active:
            # Update geometry to match the viewport size
            viewport = self._image_view._view.viewport()
            if viewport:
                self.setGeometry(0, 0, viewport.width(), viewport.height())
            self.show()
            self.raise_()  # Ensure it's on top
            # Hide the system cursor - we'll draw our own crosshair
            self._image_view._view.viewport().setCursor(Qt.BlankCursor)
        else:
            self.hide()
            self._image_view._view.viewport().unsetCursor()
            self._preview_color = None
            self.update()
    
    def is_active(self) -> bool:
        """Check if the eyedropper is active."""
        return self._active
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events for color picking."""
        if not self._active or event.button() != Qt.LeftButton:
            # Forward non-left-click events to the image view
            self._forward_event_to_image_view(event)
            return
        
        # Pick color at the mouse position
        color = self._pick_color_at_position(event.pos())
        if color is not None:
            self.color_picked.emit(color)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events for color preview."""
        if not self._active:
            # Forward events to the image view
            self._forward_event_to_image_view(event)
            return
        
        # Forward events to the image view if panning
        if self._image_view.is_panning():
            self._forward_event_to_image_view(event)
        # Update mouse position and preview color
        self._mouse_pos = event.pos()
        self._preview_color = self._pick_color_at_position(event.pos())
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        if not self._active or self._image_view.is_panning():
            # Forward events to the image view
            self._forward_event_to_image_view(event)
            return
    
    def wheelEvent(self, event) -> None:
        """Handle wheel events by forwarding to image view."""
        self._forward_wheel_event_to_image_view(event)
    
    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key_Escape and self._active:
            self.set_active(False)
            self.eyedropper_cancelled.emit()
        else:
            # Forward other key events to the image view
            self._forward_key_event_to_image_view(event)
    
    def paintEvent(self, event) -> None:
        """Paint the color preview if active."""
        if not self._active or self._preview_color is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw a small crosshair at the exact pixel being sampled
        crosshair_size = 8
        painter.setPen(QPen(Qt.white, 2))
        # Horizontal line
        painter.drawLine(self._mouse_pos.x() - crosshair_size, self._mouse_pos.y(), 
                        self._mouse_pos.x() + crosshair_size, self._mouse_pos.y())
        # Vertical line
        painter.drawLine(self._mouse_pos.x(), self._mouse_pos.y() - crosshair_size,
                        self._mouse_pos.x(), self._mouse_pos.y() + crosshair_size)
        
        # Draw a black outline for better visibility
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(self._mouse_pos.x() - crosshair_size, self._mouse_pos.y(), 
                        self._mouse_pos.x() + crosshair_size, self._mouse_pos.y())
        painter.drawLine(self._mouse_pos.x(), self._mouse_pos.y() - crosshair_size,
                        self._mouse_pos.x(), self._mouse_pos.y() + crosshair_size)
        
        # Draw color preview circle close to the cursor
        radius = 12
        offset_x = 5  # Offset to the right
        offset_y = 5  # Offset below
        
        # Ensure the preview doesn't go off-screen
        preview_x = min(self._mouse_pos.x() + offset_x, self.width() - radius * 2 - 5)
        preview_y = min(self._mouse_pos.y() + offset_y, self.height() - radius * 2 - 5)
        
        # Draw outer ring
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(self._preview_color))
        painter.drawEllipse(preview_x, preview_y, radius * 2, radius * 2)
        
        # Draw inner circle with the color
        inner_radius = radius - 3
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(preview_x + 3, preview_y + 3, inner_radius * 2, inner_radius * 2)
    
    def _pick_color_at_position(self, pos: QPoint) -> Optional[QColor]:
        """Pick color at the given position in widget coordinates."""
        # Convert widget coordinates to scene coordinates
        scene_pos = self._image_view._view.mapToScene(pos)
        
        # Get the image from app state
        display_image = self._image_view.get_current_display_image()
        if display_image is None:
            return None
        
        # Convert scene coordinates to image coordinates
        pixmap_item = self._image_view._pix_item
        if pixmap_item is None:
            return None
        
        # Get the pixmap item's position and size
        item_pos = pixmap_item.pos()
        item_size = pixmap_item.pixmap().size()
        
        # Calculate image coordinates
        x = int((scene_pos.x() - item_pos.x()) * item_size.width() / pixmap_item.pixmap().width())
        y = int((scene_pos.y() - item_pos.y()) * item_size.height() / pixmap_item.pixmap().height())
        
        # Check bounds
        if 0 <= x < display_image.shape[1] and 0 <= y < display_image.shape[0]:
            # Get the color from the image
            pixel = display_image[y, x]
            if len(pixel) >= 3:  # RGB or RGBA
                return QColor(int(pixel[0]), int(pixel[1]), int(pixel[2]))
        
        return None
    
    def _forward_event_to_image_view(self, event: QMouseEvent) -> None:
        """Forward mouse events to the image view."""
        # Create a new event with the same properties but targeted at the image view
        new_event = QMouseEvent(
            event.type(),
            event.pos(),
            event.button(),
            event.buttons(),
            event.modifiers()
        )
        self._image_view.eventFilter(self._image_view._view.viewport(), new_event)
    
    def _forward_wheel_event_to_image_view(self, event) -> None:
        """Forward wheel events to the image view."""
        self._image_view._view.wheelEvent(event)
    
    def _forward_key_event_to_image_view(self, event) -> None:
        """Forward key events to the image view."""
        self._image_view._view.keyPressEvent(event)
    
    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        # Ensure the eyedropper view covers the entire viewport
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        elif self._image_view:
            # Fallback to viewport size
            viewport = self._image_view._view.viewport()
            self.setGeometry(0, 0, viewport.width(), viewport.height())
