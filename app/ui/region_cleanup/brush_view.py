from __future__ import annotations

from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, QPoint, Signal, QPointF
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QColor, QBrush, QCursor
from PySide6.QtWidgets import QWidget

from model import AppState
from ui.image_view import ImageView


class BrushView(QWidget):
    """A transparent overlay widget for brush painting operations."""
    
    brush_painted = Signal(np.ndarray, bool)  # Emitted when image is painted (passes the modified image and is_stroke_start)
    
    def __init__(self, image_view: ImageView, app_state: AppState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_view = image_view
        self._app_state = app_state
        self._active = False
        self._brush_color = None
        self._brush_size = 24
        self._painting = False
        self._stroke_started = False  # Track if we've started a new stroke
        self._last_pos_scene: Optional[QPointF] = None
        self._mouse_pos = QPoint(0, 0)
        self._show_cursor = False
        
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setMouseTracking(True)
        # Hide the system cursor when this widget is active
        self.setCursor(Qt.BlankCursor)
        
        self.setGeometry(0, 0, self._image_view.width(), self._image_view.height())
        self.raise_()
        
        self.hide()
    
    def set_active(self, active: bool) -> None:
        """Activate or deactivate the brush view."""
        self._active = active
        if active:
            # Update geometry before showing to ensure it covers the full area
            self._update_geometry()
            self.show()
            # Hide the system cursor and set a custom cursor
            self._image_view._view.viewport().setCursor(Qt.BlankCursor)
        else:
            self.hide()
            self._image_view._view.viewport().unsetCursor()
            self._brush_color = None
            self._painting = False
            self._last_pos_scene = None
            self.update()
    
    def _update_geometry(self) -> None:
        """Update the widget geometry to match the image view."""
        if self._image_view is not None and self.parent() is not None:
            # Use parent's size to ensure we cover the full area
            parent_size = self.parent().size()
            self.setGeometry(0, 0, parent_size.width(), parent_size.height())
        elif self._image_view is not None:
            # Fallback to image view size if no parent
            view_size = self._image_view.size()
            self.setGeometry(0, 0, view_size.width(), view_size.height())
    
    def set_brush_color(self, color: QColor) -> None:
        """Set the color to use for brush painting."""
        self._brush_color = color
        self.update()
    
    def set_brush_size(self, size: int) -> None:
        """Set the brush size."""
        self._brush_size = max(1, int(size))
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events for brush painting."""
        if not self._active or self._brush_color is None:
            self._forward_event_to_image_view(event)
            return
        
        if event.button() == Qt.LeftButton:
            self._painting = True
            self._stroke_started = True  # Mark that we've started a new stroke
            # Transform widget coordinates to scene coordinates
            scene_pos = self._image_view._view.mapToScene(event.pos())
            self._last_pos_scene = scene_pos
            self._paint_brush(scene_pos)
            event.accept()
            return
        else:
            self._forward_event_to_image_view(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        self._mouse_pos = event.pos()
        
        # Always show cursor when active (within widget bounds)
        # The bounds check is only used to determine if operations are allowed
        if self._active and self._brush_color is not None:
            self._show_cursor = True
            self.update()
        
        # Handle painting while dragging
        if self._painting and event.buttons() & Qt.LeftButton:
            scene_pos = self._image_view._view.mapToScene(event.pos())
            if self._last_pos_scene is not None:
                self._paint_line(self._last_pos_scene, scene_pos)
            self._last_pos_scene = scene_pos
            event.accept()
            return
        
        self._forward_event_to_image_view(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton and self._painting:
            self._painting = False
            self._stroke_started = False  # Reset stroke flag
            self._last_pos_scene = None
            event.accept()
            return
        
        self._forward_event_to_image_view(event)
    
    def wheelEvent(self, event) -> None:
        """Handle wheel events for zooming."""
        self._forward_wheel_event_to_image_view(event)
    
    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        self._forward_key_event_to_image_view(event)
    
    def paintEvent(self, event) -> None:
        """Paint the brush cursor."""
        if not self._active or self._brush_color is None:
            return
        
        # Only show cursor preview if mouse is within widget bounds
        if not self._show_cursor:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw brush circle at mouse position
        radius = self._brush_size // 2
        center_x = self._mouse_pos.x()
        center_y = self._mouse_pos.y()
        
        # Draw outer circle (white border)
        painter.setPen(QPen(Qt.white, 2))
        painter.setBrush(QBrush(Qt.NoBrush))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw inner circle (black border)
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(center_x - radius + 1, center_y - radius + 1, (radius - 1) * 2, (radius - 1) * 2)
        
        # Draw brush color preview (semi-transparent)
        brush_color_alpha = QColor(self._brush_color)
        brush_color_alpha.setAlpha(128)
        painter.setPen(QPen(Qt.NoPen))
        painter.setBrush(QBrush(brush_color_alpha))
        painter.drawEllipse(center_x - radius + 2, center_y - radius + 2, (radius - 2) * 2, (radius - 2) * 2)
        
        # Draw center dot
        painter.setPen(QPen(Qt.black, 1))
        painter.setBrush(QBrush(self._brush_color))
        painter.drawEllipse(center_x - 2, center_y - 2, 4, 4)
    
    def _paint_brush(self, pos: QPointF) -> None:
        """Paint the brush at the given position."""
        working_image = self._app_state.working_image
        if working_image is None or self._brush_color is None:
            return
        
        # Convert scene position to image coordinates
        pixmap_item = self._image_view._pix_item
        if pixmap_item is None:
            return
        
        item_pos = pixmap_item.pos()
        item_size = pixmap_item.pixmap().size()
        
        # Convert to image coordinates
        x = int((pos.x() - item_pos.x()) * working_image.shape[1] / item_size.width())
        y = int((pos.y() - item_pos.y()) * working_image.shape[0] / item_size.height())
        
        if x < 0 or y < 0 or x >= working_image.shape[1] or y >= working_image.shape[0]:
            return
        
        # Paint a circle at the position
        radius = self._brush_size // 2
        brush_color_rgb = (self._brush_color.red(), self._brush_color.green(), self._brush_color.blue())
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < working_image.shape[1] and 0 <= py < working_image.shape[0]:
                        # Only paint on non-transparent pixels
                        if working_image.shape[2] >= 4 and working_image[py, px, 3] > 0:
                            working_image[py, px, :3] = brush_color_rgb
        
        # Emit signal with modified image and stroke start flag
        is_stroke_start = self._stroke_started
        self._stroke_started = False  # Reset after first emit
        self.brush_painted.emit(working_image, is_stroke_start)
    
    def _paint_line(self, start: QPointF, end: QPointF) -> None:
        """Paint a line between two points."""
        if self._brush_color is None:
            return
        
        # Simple line drawing algorithm
        pixmap_item = self._image_view._pix_item
        if pixmap_item is None:
            return
        
        working_image = self._app_state.working_image
        if working_image is None:
            return
        
        item_pos = pixmap_item.pos()
        item_size = pixmap_item.pixmap().size()
        
        # Convert to image coordinates
        x0 = int((start.x() - item_pos.x()) * working_image.shape[1] / item_size.width())
        y0 = int((start.y() - item_pos.y()) * working_image.shape[0] / item_size.height())
        x1 = int((end.x() - item_pos.x()) * working_image.shape[1] / item_size.width())
        y1 = int((end.y() - item_pos.y()) * working_image.shape[0] / item_size.height())
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        brush_color_rgb = (self._brush_color.red(), self._brush_color.green(), self._brush_color.blue())
        radius = self._brush_size // 2
        
        while True:
            # Paint brush at this point
            for dy_brush in range(-radius, radius + 1):
                for dx_brush in range(-radius, radius + 1):
                    if dx_brush * dx_brush + dy_brush * dy_brush <= radius * radius:
                        px, py = x0 + dx_brush, y0 + dy_brush
                        if (0 <= px < working_image.shape[1] and 
                            0 <= py < working_image.shape[0] and
                            working_image.shape[2] >= 4 and 
                            working_image[py, px, 3] > 0):
                            working_image[py, px, :3] = brush_color_rgb
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        # Emit signal with modified image (not stroke start for line painting)
        self.brush_painted.emit(working_image, False)
    
    def _forward_event_to_image_view(self, event: QMouseEvent) -> None:
        """Forward mouse events to the image view."""
        new_event = QMouseEvent(
            event.type(), event.pos(), event.button(), event.buttons(), event.modifiers()
        )
        self._image_view.eventFilter(self._image_view._view.viewport(), new_event)
    
    def _forward_wheel_event_to_image_view(self, event) -> None:
        """Forward wheel events to the image view."""
        self._image_view._view.wheelEvent(event)
    
    def _forward_key_event_to_image_view(self, event) -> None:
        """Forward key events to the image view."""
        self._image_view.keyPressEvent(event)
    
    def _is_mouse_within_image_bounds(self, pos: QPoint) -> bool:
        """Check if the mouse position is within the image bounds."""
        # Convert widget coordinates to scene coordinates
        scene_pos = self._image_view._view.mapToScene(pos)
        
        # Get the image from app state
        working_image = self._app_state.working_image
        if working_image is None:
            return False
        
        # Convert scene coordinates to image coordinates
        pixmap_item = self._image_view._pix_item
        if pixmap_item is None:
            return False
        
        # Get the pixmap item's position and size
        item_pos = pixmap_item.pos()
        item_size = pixmap_item.pixmap().size()
        
        # Calculate image coordinates
        x = int((scene_pos.x() - item_pos.x()) * working_image.shape[1] / item_size.width())
        y = int((scene_pos.y() - item_pos.y()) * working_image.shape[0] / item_size.height())
        
        # Check bounds
        return 0 <= x < working_image.shape[1] and 0 <= y < working_image.shape[0]
    
    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        self._update_geometry()
    
    def showEvent(self, event) -> None:
        """Handle show events - update geometry when shown."""
        super().showEvent(event)
        self._update_geometry()

