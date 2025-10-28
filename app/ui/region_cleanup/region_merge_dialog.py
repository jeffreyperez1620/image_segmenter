from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, 
    QGridLayout, QScrollArea, QWidget
)

from utils.qt_image import numpy_rgba_to_qimage


class ColorSwatch(QLabel):
    """A simple color swatch widget for displaying colors."""
    
    colorSelected = Signal(QColor)
    
    def __init__(self, color: QColor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.color = color
        self.setFixedSize(30, 30)
        self.setFrameStyle(1)  # Box frame
        self.setLineWidth(2)
        self.update_display()
    
    def set_color(self, color: QColor) -> None:
        """Set the color and update the display."""
        self.color = color
        self.update_display()
    
    def update_display(self) -> None:
        """Update the visual display of the color swatch."""
        # Create a pixmap with the color
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        
        # Fill with the color
        painter.fillRect(pixmap.rect(), self.color)
        
        # Draw border
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(pixmap.rect().adjusted(0, 0, -1, -1))
        
        painter.end()
        
        # Set the pixmap as the label's pixmap
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse clicks to select color."""
        if event.button() == Qt.LeftButton:
            self.colorSelected.emit(self.color)


class RegionMergeDialog(QDialog):
    """Dialog for choosing which color to merge a small region into."""
    
    def __init__(self, small_region_color: QColor, neighbor_colors: List[QColor], parent: Optional[QWidget] = None, image_data: np.ndarray = None, bbox: Tuple[int, int, int, int] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Small Region")
        self.setModal(True)
        self.resize(600, 300)
        
        self.selected_color: Optional[QColor] = None
        self.image_data = image_data
        self.bbox = bbox
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            f"The region with color RGB({small_region_color.red()}, {small_region_color.green()}, {small_region_color.blue()}) "
            f"is too small and needs to be merged. Choose which neighboring color to merge it into:"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Show image preview if available
        if self.image_data is not None and self.bbox is not None:
            preview_label = QLabel("Region Preview:")
            layout.addWidget(preview_label)
            
            # Create cropped image preview
            x, y, w, h = self.bbox
            cropped_image = self.image_data[y:y+h, x:x+w]
            
            # Convert to QImage and display - ensure array is contiguous
            cropped_image_contiguous = np.ascontiguousarray(cropped_image)
            qimg = numpy_rgba_to_qimage(cropped_image_contiguous)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale the image to a good viewing size
            # If too large, scale down; if too small, scale up
            if pixmap.width() > 400 or pixmap.height() > 400:
                pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            elif pixmap.width() < 100 or pixmap.height() < 100:
                # Scale up small regions so they're visible
                scale_factor = max(100 / pixmap.width(), 100 / pixmap.height())
                new_width = int(pixmap.width() * scale_factor)
                new_height = int(pixmap.height() * scale_factor)
                pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            preview_image_label = QLabel()
            preview_image_label.setPixmap(pixmap)
            preview_image_label.setAlignment(Qt.AlignCenter)
            preview_image_label.setStyleSheet("border: 2px solid black;")
            layout.addWidget(preview_image_label)
        
        # Show the small region color
        small_region_layout = QHBoxLayout()
        small_region_layout.addWidget(QLabel("Small region:"))
        small_swatch = ColorSwatch(small_region_color)
        small_swatch.setEnabled(False)
        small_region_layout.addWidget(small_swatch)
        small_region_layout.addStretch()
        layout.addLayout(small_region_layout)
        
        # Show neighbor colors
        neighbors_label = QLabel("Choose neighboring color to merge into:")
        layout.addWidget(neighbors_label)
        
        # Create a grid of neighbor color swatches
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        cols = 6
        for i, color in enumerate(neighbor_colors):
            swatch = ColorSwatch(color)
            swatch.colorSelected.connect(self._on_color_selected)
            row = i // cols
            col = i % cols
            scroll_layout.addWidget(swatch, row, col)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(150)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        layout.addWidget(button_box)
    
    def _on_color_selected(self, color: QColor) -> None:
        """Handle color selection."""
        self.selected_color = color
        self.ok_button.setEnabled(True)
    
    def get_selected_color(self) -> Optional[QColor]:
        """Get the selected color for merging."""
        return self.selected_color
