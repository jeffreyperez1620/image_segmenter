"""
Base class for overlay widgets that sit on top of the ImageView.

This class consolidates common functionality for overlay widgets:
- Geometry management (viewport-based sizing)
- Event forwarding to ImageView
- Active state management
- Common widget attributes and setup
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent
from PySide6.QtWidgets import QWidget

from model import AppState
from ui.image_view import ImageView


class BaseOverlayView(QWidget):
    """Base class for transparent overlay widgets on top of ImageView."""
    
    def __init__(
        self, 
        image_view: ImageView, 
        app_state: AppState, 
        parent: Optional[QWidget] = None,
        enable_mouse_tracking: bool = True,
        use_blank_cursor: bool = True
    ):
        """
        Initialize the base overlay view.
        
        Parameters
        ----------
        image_view: ImageView
            The image view this overlay sits on top of
        app_state: AppState
            Application state
        parent: Optional[QWidget]
            Parent widget (typically the viewport)
        enable_mouse_tracking: bool
            Whether to enable mouse tracking (default: True)
        use_blank_cursor: bool
            Whether to use a blank cursor when active (default: True)
        """
        super().__init__(parent)
        
        self._image_view = image_view
        self._app_state = app_state
        self._active = False
        self._enable_mouse_tracking = enable_mouse_tracking
        self._use_blank_cursor = use_blank_cursor
        
        # Make the widget transparent and able to receive mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")
        
        if enable_mouse_tracking:
            self.setMouseTracking(True)
        
        if use_blank_cursor:
            self.setCursor(Qt.BlankCursor)
        
        # Don't set geometry here - it will be set when activated
        # The ImageView might not have its final size yet during initialization
        
        # Initially hidden
        self.hide()
    
    def set_active(self, active: bool) -> None:
        """
        Activate or deactivate the overlay.
        
        Subclasses can override this to add custom activation logic,
        but should call super().set_active(active) to handle geometry and visibility.
        """
        self._active = active
        if active:
            # Update geometry to match the viewport size
            self._update_geometry()
            # Use QWidget.setVisible() directly to avoid recursion if setVisible is overridden
            QWidget.setVisible(self, True)
            self.raise_()  # Ensure it's on top
            # Hide the system cursor if using blank cursor
            if self._use_blank_cursor and self._image_view:
                self._image_view._view.viewport().setCursor(Qt.BlankCursor)
        else:
            # Use QWidget.setVisible() directly to avoid recursion if setVisible is overridden
            QWidget.setVisible(self, False)
            if self._use_blank_cursor and self._image_view:
                self._image_view._view.viewport().unsetCursor()
            self._on_deactivated()
    
    def is_active(self) -> bool:
        """Check if the overlay is active."""
        return self._active
    
    def _on_deactivated(self) -> None:
        """
        Called when the overlay is deactivated.
        
        Subclasses can override this to perform cleanup when deactivated.
        """
        pass
    
    def _update_geometry(self) -> None:
        """Update the widget geometry to match the viewport."""
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())
        elif self._image_view:
            # Fallback to viewport size
            viewport = self._image_view._view.viewport()
            if viewport:
                self.setGeometry(0, 0, viewport.width(), viewport.height())
    
    def resizeEvent(self, event) -> None:
        """Handle resize events to keep the overlay properly sized."""
        super().resizeEvent(event)
        self._update_geometry()
    
    def _forward_event_to_image_view(self, event: QMouseEvent) -> None:
        """Forward mouse events to the image view."""
        if self._image_view and hasattr(self._image_view, '_view'):
            # Create a new event with the same properties but targeted at the image view
            # Use position() for PySide6 compatibility (pos() is deprecated)
            new_event = QMouseEvent(
                event.type(),
                event.position(),
                event.button(),
                event.buttons(),
                event.modifiers()
            )
            self._image_view.eventFilter(self._image_view._view.viewport(), new_event)
    
    def _forward_wheel_event_to_image_view(self, event: QWheelEvent) -> None:
        """Forward wheel events to the image view."""
        if self._image_view and hasattr(self._image_view, '_view'):
            self._image_view._view.wheelEvent(event)
        else:
            super().wheelEvent(event)
    
    def _forward_key_event_to_image_view(self, event: QKeyEvent) -> None:
        """Forward key events to the image view."""
        if self._image_view and hasattr(self._image_view, '_view'):
            self._image_view._view.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

