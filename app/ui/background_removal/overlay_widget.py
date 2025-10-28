"""
Overlay widget for displaying include/exclude masks in background removal.

This widget handles the visual representation of user-drawn masks:
- Include areas (green overlay)
- Exclude areas (red overlay)
- Erase areas (transparent)
"""

from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor, QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem


class MaskOverlayItem(QGraphicsPixmapItem):
    """Graphics item for displaying include/exclude mask overlay."""
    
    def __init__(self):
        super().__init__()
        self.setZValue(5)  # Above image, below crop overlay
        self._mask: Optional[np.ndarray] = None
        self._image_size: Optional[tuple] = None
        
    def set_mask(self, mask: Optional[np.ndarray], image_size: tuple):
        """
        Update the mask overlay.
        
        Args:
            mask: numpy array where 0=unmarked, 1=include, 2=exclude
            image_size: (width, height) of the base image
        """
        self._mask = mask
        self._image_size = image_size
        self._update_pixmap()
    
    def _update_pixmap(self):
        """Generate and set the overlay pixmap from the mask."""
        if self._mask is None or self._image_size is None:
            self.setPixmap(QPixmap())
            return
        
        h, w = self._mask.shape
        width, height = self._image_size
        
        # Create RGBA image for overlay
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Include areas: semi-transparent green
        include_mask = (self._mask == 1)
        overlay[include_mask] = [0, 255, 0, 100]
        
        # Exclude areas: semi-transparent red
        exclude_mask = (self._mask == 2)
        overlay[exclude_mask] = [255, 0, 0, 100]
        
        # Convert to QImage
        qimage = QImage(
            overlay.data,
            w,
            h,
            w * 4,
            QImage.Format_RGBA8888
        )
        
        # Scale to match image size if needed
        if (w, h) != (width, height):
            qimage = qimage.scaled(width, height, Qt.KeepAspectRatio, Qt.FastTransformation)
        
        self.setPixmap(QPixmap.fromImage(qimage))
    
    def clear(self):
        """Clear the overlay."""
        self._mask = None
        self._image_size = None
        self.setPixmap(QPixmap())

