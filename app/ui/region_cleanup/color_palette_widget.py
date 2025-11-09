from __future__ import annotations

from typing import Optional, List
import math
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget, QGridLayout, QLabel

from utils.qt_image import numpy_rgba_to_qimage


class ColorSwatchWidget(QLabel):
    """A clickable color swatch widget."""
    
    colorSelected = Signal(QColor)
    
    def __init__(self, color: QColor, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.color = color
        self.selected = False
        self.setMinimumSize(20, 20)
        self.setMaximumSize(30, 30)
        self.setFixedSize(25, 25)  # Fixed size for consistency
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("")  # Remove default border, we'll draw it ourselves
        self.update_display()
    
    def set_selected(self, selected: bool):
        """Set the selected state."""
        self.selected = selected
        self.update_display()
    
    def update_display(self):
        """Update the visual display of the color swatch."""
        # Create a pixmap with the color
        pixmap = self._create_color_pixmap()
        self.setPixmap(pixmap)
    
    def _create_color_pixmap(self):
        """Create a pixmap showing the color."""
        # Use fixed size for consistency
        pixmap = self._create_pixmap(25, 25)
        return pixmap
    
    def _create_pixmap(self, width: int, height: int):
        """Create a pixmap of the specified size."""
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(width, height)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill with the color
        painter.fillRect(pixmap.rect(), self.color)
        
        # Draw border
        if self.selected:
            # Distinctive border for selected: thick white outer border with black inner border
            # Draw outer white border (thick)
            painter.setPen(QPen(Qt.white, 4))
            painter.drawRect(pixmap.rect().adjusted(1, 1, -2, -2))
            # Draw inner black border for contrast
            painter.setPen(QPen(Qt.black, 2))
            painter.drawRect(pixmap.rect().adjusted(3, 3, -4, -4))
        else:
            # Simple thin border for unselected
            painter.setPen(QPen(Qt.black, 1))
            painter.drawRect(pixmap.rect().adjusted(0, 0, -1, -1))
        
        painter.end()
        return pixmap
    
    def mousePressEvent(self, event):
        """Handle mouse clicks to select color."""
        if event.button() == Qt.LeftButton:
            self.colorSelected.emit(self.color)
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_display()


class ColorPaletteWidget(QWidget):
    """Widget displaying a grid of color swatches."""
    
    colorSelected = Signal(QColor)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._colors: List[QColor] = []
        self._selected_color: Optional[QColor] = None
        self._swatches: List[ColorSwatchWidget] = []
        
        self._layout = QGridLayout(self)
        self._layout.setSpacing(4)
        self._layout.setContentsMargins(4, 4, 4, 4)
    
    def set_colors(self, colors: List[QColor]):
        """Set the colors to display in the palette."""
        self._colors = colors
        
        # Clear existing swatches
        for swatch in self._swatches:
            swatch.deleteLater()
        self._swatches.clear()
        
        if not colors:
            return
        
        # Calculate grid dimensions (as square as possible)
        num_colors = len(colors)
        cols = int(math.ceil(math.sqrt(num_colors)))
        rows = int(math.ceil(num_colors / cols))
        
        # Create swatches
        for i, color in enumerate(colors):
            swatch = ColorSwatchWidget(color, self)
            swatch.colorSelected.connect(self._on_swatch_selected)
            self._swatches.append(swatch)
            
            row = i // cols
            col = i % cols
            self._layout.addWidget(swatch, row, col)
        
        # If we have a previously selected color, try to maintain selection
        if self._selected_color:
            self._update_selection(self._selected_color)
    
    def set_selected_color(self, color: QColor):
        """Set the selected color."""
        self._selected_color = color
        self._update_selection(color)
    
    def get_selected_color(self) -> Optional[QColor]:
        """Get the currently selected color."""
        return self._selected_color
    
    def _on_swatch_selected(self, color: QColor):
        """Handle color swatch selection."""
        self._selected_color = color
        self._update_selection(color)
        self.colorSelected.emit(color)
    
    def _update_selection(self, color: QColor):
        """Update which swatch is selected."""
        for swatch in self._swatches:
            # Compare colors (RGB only, ignore alpha)
            swatch_color = swatch.color
            if (swatch_color.red() == color.red() and 
                swatch_color.green() == color.green() and 
                swatch_color.blue() == color.blue()):
                swatch.set_selected(True)
            else:
                swatch.set_selected(False)

