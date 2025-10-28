"""
Background removal view for handling background removal specific interactions.

This widget handles:
- Include/exclude mask painting
- Crop rectangle interaction
- Undo/redo for background removal operations
- Mask overlay display
"""

from typing import Optional, List
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QColor, QPixmap, QImage
from PySide6.QtWidgets import QWidget, QGraphicsView

from model import AppState
from ui.image_view import ImageView

from .overlay_widget import MaskOverlayItem
from .crop_rectangle_widget import CropRectangleItem


class BackgroundRemovalView(QWidget):
    """Widget for handling background removal specific interactions."""
    
    # Signals
    mask_changed = Signal(np.ndarray)  # Emitted when mask is modified
    crop_rect_changed = Signal(QRectF)  # Emitted when crop rectangle changes
    
    def __init__(self, parent=None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None):
        super().__init__(parent)
        
        # Make this widget transparent and able to receive mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")
        
        self._image_view = image_view
        self._app_state = app_state
        
        # Interaction state
        self._mode: str = "include"  # include|exclude|erase|crop
        self._brush_size: int = 24
        self._painting: bool = False
        self._last_pos_scene: Optional[QPointF] = None
        self._crop_origin: Optional[QPointF] = None
        self._crop_rect: Optional[QRectF] = None
        self._cropping_active: bool = False
        
        # Masks: 0=unmarked, 1=include, 2=exclude
        self._user_mask: Optional[np.ndarray] = None
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        
        # Overlay items
        self._mask_overlay = MaskOverlayItem()
        self._crop_overlay = CropRectangleItem()
        
        # Add overlay items to the image view's scene if available
        if self._image_view and hasattr(self._image_view, '_scene'):
            self._image_view._scene.addItem(self._mask_overlay)
            self._image_view._scene.addItem(self._crop_overlay)
            self._mask_overlay.setZValue(5)  # Above image, below crop
            self._crop_overlay.setZValue(10)  # Above everything
        
        # Image state
        self._image_size: Optional[tuple] = None
		
		# Position the background removal view over the image view
        self.setGeometry(0, 0, self._image_view.width(), self._image_view.height())
        self.raise_()
        
    def set_image_size(self, width: int, height: int):
        """Set the size of the image for mask operations."""
        self._image_size = (width, height)
        self._init_masks()
        
    def _init_masks(self):
        """Initialize the user mask."""
        if self._image_size is None:
            return
        width, height = self._image_size
        self._user_mask = np.zeros((height, width), dtype=np.uint8)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_mask_overlay()
        
    def set_mode(self, mode: str):
        """Set the interaction mode."""
        self._mode = mode
        
    def hide_overlays(self):
        """Hide all overlays when switching tabs."""
        self._mask_overlay.setVisible(False)
        self._crop_overlay.setVisible(False)
            
    def show_overlays(self):
        """Show overlays when returning to this tab."""
        self._mask_overlay.setVisible(True)
        self._crop_overlay.setVisible(True)
        
    def resizeEvent(self, event):
        """Handle resize events to keep the overlay properly sized."""
        super().resizeEvent(event)
        # The overlay should always match the parent's size
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        
    def set_brush_size(self, size: int):
        """Set the brush size for painting."""
        self._brush_size = max(1, int(size))
        
    def get_user_mask(self) -> Optional[np.ndarray]:
        """Get the current user mask."""
        return self._user_mask
        
    def get_crop_rect_xywh(self) -> Optional[tuple]:
        """Get the crop rectangle as (x, y, width, height)."""
        if self._crop_rect is None:
            return None
        return (
            int(self._crop_rect.x()),
            int(self._crop_rect.y()),
            int(self._crop_rect.width()),
            int(self._crop_rect.height())
        )
        
    def set_crop_rect(self, rect: Optional[QRectF]):
        """Set the crop rectangle."""
        self._crop_rect = rect
        self._update_crop_overlay()
        if rect is not None:
            self.crop_rect_changed.emit(rect)
            
    def clear_marks(self):
        """Clear all marks and reset state."""
        self._init_masks()
        self._crop_rect = None
        self._crop_origin = None
        self._cropping_active = False
        self._update_mask_overlay()
        self._update_crop_overlay()
        
    def undo(self):
        """Undo the last mask operation."""
        if self._undo_stack:
            if self._user_mask is not None:
                self._redo_stack.append(self._user_mask.copy())
            self._user_mask = self._undo_stack.pop()
            self._update_mask_overlay()
            self.mask_changed.emit(self._user_mask)
            
    def redo(self):
        """Redo the last undone mask operation."""
        if self._redo_stack:
            if self._user_mask is not None:
                self._undo_stack.append(self._user_mask.copy())
            self._user_mask = self._redo_stack.pop()
            self._update_mask_overlay()
            self.mask_changed.emit(self._user_mask)
            
    def _update_mask_overlay(self):
        """Update the mask overlay display."""
        if self._user_mask is not None and self._image_size is not None:
            self._mask_overlay.set_mask(self._user_mask, self._image_size)
            
    def _update_crop_overlay(self):
        """Update the crop rectangle overlay."""
        if self._image_size is not None:
            width, height = self._image_size
            image_bounds = QRectF(0, 0, width, height)
            self._crop_overlay.set_crop_rect(self._crop_rect, image_bounds)
            
    def _save_mask_state(self):
        """Save current mask state to undo stack."""
        if self._user_mask is not None:
            self._undo_stack.append(self._user_mask.copy())
            # Limit undo stack size
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            # Clear redo stack when new action is performed
            self._redo_stack.clear()
            
    def _paint_mask(self, pos: QPointF, value: int):
        """Paint the mask at the given position."""
        if self._user_mask is None or self._image_size is None:
            return
            
        # Convert scene position to image coordinates
        x = int(pos.x())
        y = int(pos.y())
        
        if x < 0 or y < 0 or x >= self._image_size[0] or y >= self._image_size[1]:
            return
            
        # Paint a circle at the position
        radius = self._brush_size // 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < self._image_size[0] and 0 <= py < self._image_size[1]:
                        self._user_mask[py, px] = value
                        
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        
        # Only handle left mouse button events for painting/cropping
        if event.button() == Qt.LeftButton:
            if self._mode in ["include", "exclude", "erase"]:
                self._painting = True
                # Transform widget coordinates to scene coordinates
                scene_pos = self._image_view._view.mapToScene(event.position().toPoint())
                self._last_pos_scene = scene_pos
                self._save_mask_state()
                
                value = 1 if self._mode == "include" else (2 if self._mode == "exclude" else 0)
                self._paint_mask(scene_pos, value)
                self._update_mask_overlay()
                self.mask_changed.emit(self._user_mask)
                event.accept()
                return
                
            elif self._mode == "crop":
                self._cropping_active = True
                # Transform widget coordinates to scene coordinates
                self._crop_origin = self._image_view._view.mapToScene(event.position().toPoint())
                event.accept()
                return
        
        # For all other mouse events (right-click, middle-click, etc.), forward them to ImageView
        # Create a new event and forward it to the ImageView's event filter
        new_event = QMouseEvent(
            event.type(),
            event.position(),
            event.button(),
            event.buttons(),
            event.modifiers()
        )
        self._image_view.eventFilter(self._image_view._view.viewport(), new_event)
        
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        # Only handle left mouse button events for painting/cropping
        if event.buttons() & Qt.LeftButton:
            if self._mode in ["include", "exclude", "erase"] and self._painting:
                if self._last_pos_scene is not None:
                    # Transform widget coordinates to scene coordinates
                    scene_pos = self._image_view._view.mapToScene(event.position().toPoint())
                    # Interpolate between last position and current position
                    self._paint_line(self._last_pos_scene, scene_pos)
                    self._last_pos_scene = scene_pos
                    self._update_mask_overlay()
                    self.mask_changed.emit(self._user_mask)
                event.accept()
                return
                
            elif self._mode == "crop" and self._cropping_active:
                if self._crop_origin is not None:
                    # Transform widget coordinates to scene coordinates
                    scene_pos = self._image_view._view.mapToScene(event.position().toPoint())
                    current_rect = QRectF(self._crop_origin, scene_pos).normalized()
                    self._crop_overlay.set_crop_rect(current_rect, QRectF(0, 0, *self._image_size) if self._image_size else QRectF())
                event.accept()
                return
        
        # For all other mouse move events, forward them to ImageView
        # Create a new event and forward it to the ImageView's event filter
        new_event = QMouseEvent(
            event.type(),
            event.position(),
            event.button(),
            event.buttons(),
            event.modifiers()
        )
        self._image_view.eventFilter(self._image_view._view.viewport(), new_event)
        
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        # Only handle left mouse button events for painting/cropping
        if event.button() == Qt.LeftButton:
            if self._mode in ["include", "exclude", "erase"] and self._painting:
                self._painting = False
                self._last_pos_scene = None
                event.accept()
                return
                
            elif self._mode == "crop" and self._cropping_active:
                if self._crop_origin is not None:
                    # Transform widget coordinates to scene coordinates
                    scene_pos = self._image_view._view.mapToScene(event.position().toPoint())
                    final_rect = QRectF(self._crop_origin, scene_pos).normalized()
                    self.set_crop_rect(final_rect)
                self._cropping_active = False
                self._crop_origin = None
                event.accept()
                return
        
        # For all other mouse release events, forward them to ImageView
        # Create a new event and forward it to the ImageView's event filter
        new_event = QMouseEvent(
            event.type(),
            event.position(),
            event.button(),
            event.buttons(),
            event.modifiers()
        )
        self._image_view.eventFilter(self._image_view._view.viewport(), new_event)
    
    def wheelEvent(self, event) -> None:
        """Handle wheel events for zooming."""
        # Forward all wheel events to the ImageView for zoom functionality
        # Forward the event to the QGraphicsView inside ImageView
        self._image_view._view.wheelEvent(event)
        
    def _paint_line(self, start: QPointF, end: QPointF):
        """Paint a line between two points."""
        if self._user_mask is None:
            return
            
        value = 1 if self._mode == "include" else (2 if self._mode == "exclude" else 0)
        
        # Simple line drawing algorithm
        x0, y0 = int(start.x()), int(start.y())
        x1, y1 = int(end.x()), int(end.y())
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            self._paint_mask(QPointF(x0, y0), value)
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def apply_crop(self):
        """Apply the crop rectangle to the image."""
        if self._crop_rect is None:
            return
        
        # Get the working image from app state
        working_image = self._app_state.working_image
        if working_image is None:
            return
        
        # Ensure we have a numpy RGBA array
        from utils.qt_image import ensure_numpy_rgba
        rgba_image = ensure_numpy_rgba(working_image)
        
        h, w = rgba_image.shape[:2]
        
        # Get crop rectangle coordinates
        x = int(self._crop_rect.x())
        y = int(self._crop_rect.y())
        crop_w = int(self._crop_rect.width())
        crop_h = int(self._crop_rect.height())
        
        # Clamp coordinates to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        crop_w = max(1, min(crop_w, w - x))
        crop_h = max(1, min(crop_h, h - y))
        
        # Crop the RGBA image
        cropped_rgba = rgba_image[y:y+crop_h, x:x+crop_w]
        
        # Update the working image with the cropped RGBA array
        self._app_state.working_image = cropped_rgba
        self._image_size = (crop_w, crop_h)
        self._init_masks()
        self.clear_marks()
