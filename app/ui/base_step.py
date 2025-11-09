"""
Base class for all processing steps that provides common base/working image functionality.
"""

from tabnanny import check
from typing import Optional, Callable
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap

from model import AppState
from ui.image_view import ImageView
from utils.qt_image import numpy_rgba_to_qimage, qimage_to_numpy_rgba


class BaseStep(QWidget):
    """
    Base class for all processing steps that provides common base/working image functionality.
    
    This class provides:
    - Show Base Image checkbox
    - Apply to Base button  
    - Reset button
    - Common methods for accessing base/working images
    - Signal for when base/working images change
    - Centralized image management for all steps
    """
    
    def __init__(self, parent=None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None):
        super().__init__(parent)
        self._app_state = app_state
        self._image_view = image_view
        self._has_unapplied_changes = False
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the common UI elements."""
        layout = QVBoxLayout(self)
        
        # Create the main widget container for step-specific content
        self._main_widget = QWidget()
        self._main_layout = QVBoxLayout(self._main_widget)
        layout.addWidget(self._main_widget)
        
        # Add common controls at the bottom
        self._init_common_controls(layout)
    
    def _init_common_controls(self, parent_layout):
        """Initialize the common base/working image controls."""
        # Base/Working image controls
        controls_layout = QHBoxLayout()
        
        # Show Base Image checkbox
        self._chk_show_base = QCheckBox("Show Base Image")
        controls_layout.addWidget(self._chk_show_base)
        
        # Apply to Base button
        self._btn_apply_to_base = QPushButton("Apply to Base")
        controls_layout.addWidget(self._btn_apply_to_base)
        
        # Reset button
        self._btn_reset = QPushButton("Reset")
        controls_layout.addWidget(self._btn_reset)
        
        # Add stretch to push controls to the right
        controls_layout.addStretch()
        
        parent_layout.addLayout(controls_layout)
    
    def set_main_widget(self, widget: QWidget):
        """Set the main widget for this step."""
        from PySide6.QtWidgets import QSizePolicy
        
        # Clear existing main widget
        if hasattr(self, '_main_widget'):
            self._main_widget.deleteLater()
        
        self._main_widget = widget
        # Set size policy to allow expansion
        self._main_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        # Only create a layout if the widget doesn't already have one
        if self._main_widget.layout() is None:
            self._main_layout = QVBoxLayout(self._main_widget)
        else:
            self._main_layout = self._main_widget.layout()
        
        # Ensure we have a layout before trying to insert
        if self.layout() is not None:
            self.layout().insertWidget(0, self._main_widget)
        else:
            # If no layout exists, create one
            layout = QVBoxLayout(self)
            layout.addWidget(self._main_widget)
            self._init_common_controls(layout)
    
    def enable_main_widget_stretch(self):
        """Enable stretching of the main widget to push controls to the bottom."""
        main_layout = self.layout()
        if main_layout is not None:
            # The main widget should be at index 0, controls layout should be at the end
            # We want to add a stretch at index 1 (after main widget, before controls)
            item_count = main_layout.count()
            if item_count >= 2:
                # Check if there's already a stretch at index 1
                item = main_layout.itemAt(1)
                # If item doesn't exist or is not a spacer, add a stretch
                if item is None:
                    # No item at index 1, add a stretch
                    main_layout.insertStretch(1, 1)
                elif item.spacerItem() is None:
                    # Item exists but is not a spacer, insert stretch before it
                    main_layout.insertStretch(1, 1)
    
    def set_base_image(self, image: np.ndarray):
        """Set the base image."""
        self._app_state.base_image = image
    
    def set_working_image(self, image: Optional[np.ndarray]):
        """Set the working image."""
        self._app_state.working_image = image
        self.mark_changes_unapplied()
        self._chk_show_base.setChecked(False)
    
    def get_base_image(self) -> Optional[np.ndarray]:
        """Get the base image."""
        return self._app_state.base_image
    
    def get_working_image(self) -> Optional[np.ndarray]:
        """Get the working image."""
        return self._app_state.working_image
    
    def get_current_display_image(self) -> Optional[np.ndarray]:
        """Get the image that should be displayed (base or working based on checkbox)."""
        if self._chk_show_base.isChecked():
            return self._app_state.base_image
        else:
            return self._app_state.working_image
    
    def mark_changes_applied(self):
        """Mark that changes have been applied or reset (working becomes base)."""
        self._has_unapplied_changes = False
        self._update_controls_state()
    
    def mark_changes_unapplied(self):
        """Mark that there are unapplied changes."""
        self._has_unapplied_changes = True
        self._update_controls_state()

    def has_unapplied_changes(self) -> bool:
        """Check if there are unapplied changes."""
        return self._has_unapplied_changes

    def validate_entry(self) -> bool:
        """Validate the entry of the step."""
        return True
    
    def _update_controls_state(self):
        """Update the state of common controls."""
        has_base = self._app_state.base_image is not None
        has_working = self._app_state.working_image is not None
        
        # Enable/disable controls based on available images
        self._chk_show_base.setEnabled(has_base)
        self._btn_apply_to_base.setEnabled(has_working and self._has_unapplied_changes)
        self._btn_reset.setEnabled(has_working and self._has_unapplied_changes)
    
    def _on_show_base_toggled(self, checked: bool):
        """Handle Show Base Image checkbox toggle."""
        self._image_view.set_show_working(not checked)
    
    def _on_apply_to_base_clicked(self):
        """Handle Apply to Base button click."""
        self._app_state.apply_working_to_base()
        self.mark_changes_applied()
    
    def _on_reset_clicked(self):
        """Handle Reset button click."""
        if self._app_state.base_image is not None and self._app_state.working_image is not None:
            # Show confirmation dialog
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Reset Changes",
                "This will reset your working image to the base image, discarding all changes.\n\nDo you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._app_state.reset_working_image()
    
    def get_display_pixmap(self) -> Optional[QPixmap]:
        """Get the pixmap that should be displayed."""
        image = self.get_current_display_image()
        if image is not None:
            qimg = numpy_rgba_to_qimage(image)
            return QPixmap.fromImage(qimg)
        return None
    
    def _on_open(self) -> None:
        """
        Called when this step becomes active.
        Subclasses can override this to connect widgets, initialize state, etc.
        """
        self._chk_show_base.toggled.connect(self._on_show_base_toggled)
        self._btn_apply_to_base.clicked.connect(self._on_apply_to_base_clicked)
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        self._chk_show_base.setChecked(False)
        if self._image_view is not None:
            self._image_view.set_show_working(True)
    
    def _on_close(self) -> None:
        """
        Called when this step becomes inactive.
        Subclasses can override this to disconnect widgets, cleanup state, etc.
        """
        self._chk_show_base.toggled.disconnect(self._on_show_base_toggled)
        self._btn_apply_to_base.clicked.disconnect(self._on_apply_to_base_clicked)
        self._btn_reset.clicked.disconnect(self._on_reset_clicked)

    def _update_controls(self) -> None:
        """Update the Show Base Image checkbox state."""
        imagesDifferent = self._app_state.base_image != self._app_state.working_image
        
        self._btn_apply_to_base.setEnabled(imagesDifferent)
        self._btn_reset.setEnabled(imagesDifferent)