from __future__ import annotations

from typing import Optional, List
import numpy as np
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen, QIcon, QPainterPath, QBrush
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, 
    QGridLayout, QScrollArea, QWidget, QPushButton
)

from utils.qt_image import numpy_rgba_to_qimage


class ColorSwatch(QLabel):
    """A color swatch widget for displaying and selecting colors."""
    
    colorSelected = Signal(QColor)
    
    def __init__(self, color: QColor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.color = color
        self.setFixedSize(40, 40)
        self.setFrameStyle(1)  # Box frame
        self.setLineWidth(2)
        self.setCursor(Qt.PointingHandCursor)
        self.update_display()
    
    def set_color(self, color: QColor) -> None:
        """Set the color and update the display."""
        self.color = color
        self.update_display()
    
    def update_display(self) -> None:
        """Update the visual display of the color swatch."""
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        
        # Fill with the color
        painter.fillRect(pixmap.rect(), self.color)
        
        # Draw border
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(pixmap.rect().adjusted(1, 1, -1, -1))
        
        painter.end()
        
        # Set the pixmap as the label's pixmap
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """Handle mouse clicks to select color."""
        if event.button() == Qt.LeftButton:
            self.colorSelected.emit(self.color)


class FloodFillColorDialog(QDialog):
    """Dialog for selecting a flood fill color from image colors."""
    
    def __init__(self, image_colors: List[QColor], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Flood Fill Color")
        self.setModal(True)
        self.resize(500, 400)
        
        self.selected_color: Optional[QColor] = None
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Choose a color to use for flood fill:")
        instructions.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(instructions)
        
        # Create a scrollable area for the color swatches
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        # Arrange colors in a grid
        cols = 8
        for i, color in enumerate(image_colors):
            swatch = ColorSwatch(color)
            swatch.colorSelected.connect(self._on_color_selected)
            row = i // cols
            col = i % cols
            scroll_layout.addWidget(swatch, row, col)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _on_color_selected(self, color: QColor) -> None:
        """Handle color selection."""
        self.selected_color = color
        self.accept()  # Close dialog immediately when color is selected
    
    def get_selected_color(self) -> Optional[QColor]:
        """Get the selected color for flood fill."""
        return self.selected_color


def create_flood_fill_icon(color: QColor = QColor(0, 0, 0)) -> QIcon:
    """Create a flood fill icon with the specified color."""
    # Create a 32x32 pixmap for the icon
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Draw the flood fill icon (bucket shape)
    # Main bucket body
    bucket_rect = QRect(8, 12, 16, 16)
    painter.setBrush(QBrush(color))
    painter.setPen(QPen(Qt.black, 2))
    painter.drawRect(bucket_rect)
    
    # Bucket handle
    handle_path = QPainterPath()
    handle_path.moveTo(24, 12)
    handle_path.quadTo(28, 8, 24, 4)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(Qt.black, 2))
    painter.drawPath(handle_path)
    
    # Add some "paint drops" to indicate flood fill
    painter.setBrush(QBrush(color))
    painter.setPen(QPen(Qt.black, 1))
    painter.drawEllipse(10, 20, 3, 3)
    painter.drawEllipse(14, 22, 2, 2)
    painter.drawEllipse(18, 21, 2, 2)
    
    painter.end()
    
    return QIcon(pixmap)
