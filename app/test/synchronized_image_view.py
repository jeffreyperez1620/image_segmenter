#!/usr/bin/env python3
"""
Synchronized image view component for the smoothing test application.
"""

import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QPainter


class SynchronizedImageView(QGraphicsView):
    """Custom image view that can be synchronized with other views."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Synchronization
        self.synchronized_views = []
        self.is_syncing = False
        
        # Scene setup
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Image item
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        
        # Mouse tracking for panning
        self.last_pan_point = QPointF()
        self.panning = False

    def set_pixmap(self, pixmap: QPixmap):
        """Set the image pixmap."""
        self.image_item.setPixmap(pixmap)
        self.fit_in_view()

    def fit_in_view(self):
        """Fit the image to the view."""
        if self.image_item.pixmap().isNull():
            return
        self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def add_synchronized_view(self, view):
        """Add a view to synchronize with this one."""
        if view not in self.synchronized_views:
            self.synchronized_views.append(view)

    def wheelEvent(self, event: QWheelEvent):
        """Handle wheel events for zooming with synchronization."""
        if self.is_syncing:
            return
            
        # Get the zoom factor
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:
            zoom_factor = 1.0 / zoom_factor
        
        # Apply zoom
        self.scale(zoom_factor, zoom_factor)
        
        # Sync with other views
        self._sync_zoom()
        
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning."""
        if event.button() == Qt.LeftButton:
            self.last_pan_point = event.position()
            self.panning = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.panning = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mouseReleaseEvent(event)

    def _sync_zoom(self):
        """Synchronize zoom with other views."""
        if self.is_syncing:
            return
            
        # Get current transform
        transform = self.transform()
        
        # Apply to synchronized views
        for view in self.synchronized_views:
            if view != self:
                view.is_syncing = True
                view.setTransform(transform)
                view.is_syncing = False

    def _sync_pan(self):
        """Synchronize pan with other views."""
        if self.is_syncing:
            return
            
        # Get current scroll positions
        h_scroll = self.horizontalScrollBar().value()
        v_scroll = self.verticalScrollBar().value()
        
        # Apply to synchronized views
        for view in self.synchronized_views:
            if view != self:
                view.is_syncing = True
                view.horizontalScrollBar().setValue(h_scroll)
                view.verticalScrollBar().setValue(v_scroll)
                view.is_syncing = False

    def sync_from_view(self, source_view):
        """Synchronize this view with a source view."""
        if self.is_syncing:
            return
            
        self.is_syncing = True
        
        # Copy transform
        self.setTransform(source_view.transform())
        
        # Copy scroll positions
        self.horizontalScrollBar().setValue(source_view.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(source_view.verticalScrollBar().value())
        
        self.is_syncing = False




