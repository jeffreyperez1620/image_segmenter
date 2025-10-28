"""
Crop rectangle widget for background removal.

This widget handles the visual representation and interaction with the crop rectangle:
- Drawing the crop rectangle
- Showing the semi-transparent overlay outside the crop area
- Handling crop rectangle updates
"""

from typing import Optional
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainterPath, QColor
from PySide6.QtWidgets import QGraphicsPathItem


class CropRectangleItem(QGraphicsPathItem):
    """Graphics item for displaying crop rectangle with overlay."""
    
    def __init__(self):
        super().__init__()
        self.setZValue(10)  # Above everything else
        self.setBrush(QColor(255, 105, 180, 90))  # Semi-transparent pink
        self.setPen(Qt.NoPen)
        self.setVisible(False)
        self._crop_rect: Optional[QRectF] = None
        self._image_bounds: Optional[QRectF] = None
    
    def set_crop_rect(self, crop_rect: Optional[QRectF], image_bounds: QRectF):
        """
        Update the crop rectangle.
        
        Args:
            crop_rect: The crop rectangle in scene coordinates, or None to clear
            image_bounds: The bounds of the full image in scene coordinates
        """
        self._crop_rect = crop_rect
        self._image_bounds = image_bounds
        self._update_path()
    
    def _update_path(self):
        """Generate the path that shows the overlay outside the crop area."""
        if self._crop_rect is None or self._image_bounds is None:
            self.setPath(QPainterPath())
            self.setVisible(False)
            return
        
        # Create path that covers everything except the crop rectangle
        path = QPainterPath()
        path.addRect(self._image_bounds)
        
        # Subtract the crop rectangle
        crop_path = QPainterPath()
        crop_path.addRect(self._crop_rect)
        path = path.subtracted(crop_path)
        
        self.setPath(path)
        self.setVisible(True)
    
    def clear(self):
        """Clear the crop rectangle."""
        self._crop_rect = None
        self._image_bounds = None
        self.setPath(QPainterPath())
        self.setVisible(False)
    
    def get_crop_rect(self) -> Optional[QRectF]:
        """Get the current crop rectangle."""
        return self._crop_rect

