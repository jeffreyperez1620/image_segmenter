"""
Graphics item for displaying individual regions in the Arrange Regions step.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PySide6.QtWidgets import QGraphicsPixmapItem, QStyleOptionGraphicsItem, QWidget

from processing.arrange_regions import RegionData
from utils.qt_image import numpy_rgba_to_qimage


class RegionPixmapItem(QGraphicsPixmapItem):
    """Graphics item representing a single draggable region."""
    
    def __init__(self, region_data: RegionData):
        super().__init__()
        self._region_data = region_data
        self._hovered = False
        
        # Convert region image to pixmap
        qimg = numpy_rgba_to_qimage(region_data.image)
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(pixmap)
        
        # Set position
        self.setPos(region_data.position[0], region_data.position[1])
        
        # Set Z-value to be above base image
        self.setZValue(100)
    
    def get_region_data(self) -> RegionData:
        """Get the region data."""
        return self._region_data
    
    def set_hovered(self, hovered: bool) -> None:
        """Set hovered state."""
        if self._hovered != hovered:
            self._hovered = hovered
            self.update()
    
    def update_position(self) -> None:
        """Update position from region data."""
        pos = self.pos()
        self._region_data.position = (int(pos.x()), int(pos.y()))
        self.update()
    
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        """Paint the region with outline if hovered."""
        # Paint the pixmap
        super().paint(painter, option, widget)
        
        # Draw outline if hovered
        if self._hovered:
            pen = QPen(QColor(255, 255, 0), 2)  # Yellow outline
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.boundingRect())

