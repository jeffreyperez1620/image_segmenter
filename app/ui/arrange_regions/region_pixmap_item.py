"""
Graphics item for displaying individual regions in the Arrange Regions step.
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np
import cv2 as cv
from PySide6.QtCore import Qt, QPointF, QRectF, QPoint
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QImage, QPainterPath
from PySide6.QtWidgets import QGraphicsPixmapItem, QStyleOptionGraphicsItem, QWidget

from processing.arrange_regions import RegionData
from utils.qt_image import numpy_rgba_to_qimage


class RegionPixmapItem(QGraphicsPixmapItem):
    """Graphics item representing a single draggable region."""
    
    def __init__(self, region_data: RegionData):
        super().__init__()
        self._region_data = region_data
        self._hovered = False
        self._active = False
        self._contour_path: Optional[QPainterPath] = None
        
        # Convert region image to pixmap
        qimg = numpy_rgba_to_qimage(region_data.image)
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(pixmap)
        
        # Extract contour from the region image
        self._extract_contour(region_data.image)
        
        # Calculate centroid for transform origin (before setting pixmap position)
        # We need to get the centroid in item coordinates (relative to pixmap)
        pixmap = self.pixmap()
        if not pixmap.isNull():
            centroid_item = QPointF(pixmap.width() / 2.0, pixmap.height() / 2.0)
        else:
            centroid_item = QPointF(0, 0)
        
        # Set transform origin to centroid so rotation happens about the center
        self.setTransformOriginPoint(centroid_item)
        
        # Set position - the stored position is top-left, so we set it directly
        # The transform origin will handle rotation about the centroid
        self.setPos(region_data.position[0], region_data.position[1])
        
        # Set rotation
        self.setRotation(region_data.rotation)
        
        # Set Z-value to be above base image
        self.setZValue(100)
        
        # Disable mouse event acceptance - the overlay widget will handle all mouse interactions
        # This prevents the graphics items from capturing events before the overlay receives them
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setAcceptHoverEvents(False)
    
    def _extract_contour(self, region_image: np.ndarray) -> None:
        """Extract the contour path from the region image."""
        # Get alpha channel as mask
        if region_image.shape[2] == 4:
            alpha = region_image[:, :, 3]
        else:
            # If no alpha, assume all pixels are part of the region
            alpha = np.ones((region_image.shape[0], region_image.shape[1]), dtype=np.uint8) * 255
        
        # Create binary mask
        mask = (alpha > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback to bounding rect if no contours found
            self._contour_path = None
            return
        
        # Create QPainterPath from contours
        path = QPainterPath()
        for contour in contours:
            if len(contour) < 3:
                continue
            
            # Convert contour points to QPointF
            points = [QPointF(float(pt[0][0]), float(pt[0][1])) for pt in contour]
            
            # Start the path
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        
        self._contour_path = path
    
    def get_region_data(self) -> RegionData:
        """Get the region data."""
        return self._region_data
    
    def set_hovered(self, hovered: bool) -> None:
        """Set hovered state."""
        if self._hovered != hovered:
            self._hovered = hovered
            self.update()
    
    def set_active(self, active: bool) -> None:
        """Set active state (shows outline and rotation handle)."""
        if self._active != active:
            self._active = active
            self.update()
    
    def is_active(self) -> bool:
        """Check if this item is active."""
        return self._active
    
    def get_centroid(self) -> QPointF:
        """Get the centroid of the region in item coordinates."""
        # Use the transform origin point as the centroid (it's set to the pixmap center)
        return self.transformOriginPoint()
    
    def update_position(self) -> None:
        """Update position from region data."""
        # The position is stored as top-left, and that's what setPos uses
        # So we can just read it directly
        pos = self.pos()
        self._region_data.position = (int(pos.x()), int(pos.y()))
        self.update()
    
    def update_rotation(self) -> None:
        """Update rotation from region data."""
        self.setRotation(self._region_data.rotation)
        self.update()
    
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        """Paint the region with outline if hovered or active."""
        # Paint the pixmap
        super().paint(painter, option, widget)
        
        # Draw contour outline if hovered or active
        if (self._hovered or self._active) and self._contour_path is not None:
            pen = QPen(QColor(255, 255, 0), 2)  # Yellow outline
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self._contour_path)
        elif self._hovered or self._active:
            # Fallback to bounding rect if no contour available
            pen = QPen(QColor(255, 255, 0), 2)  # Yellow outline
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.boundingRect())

