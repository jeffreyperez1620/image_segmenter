#!/usr/bin/env python3
"""
Visualization components for the smoothing test application.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy,
    QFileDialog, QMessageBox, QCheckBox, QGraphicsTextItem, QGraphicsRectItem
)
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor, QMouseEvent, QBrush, QPen

# Add the app directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.qt_image import numpy_rgba_to_qimage
from synchronized_image_view import SynchronizedImageView


class PixelCalloutImageView(SynchronizedImageView):
    """Custom image view with pixel color callout functionality."""
    
    def __init__(self, original_image, parent=None):
        super().__init__(parent)
        self.original_image = original_image
        self.pixel_callout_enabled = False
        
        # Enable mouse tracking for hover events
        self.setMouseTracking(True)
        
        # Create callout label as a widget overlay
        self.callout_label = QLabel(self)
        self.callout_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: white;
                border: 2px solid rgba(255, 255, 255, 200);
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        self.callout_label.hide()
    
    def set_pixel_callout_enabled(self, enabled):
        """Enable or disable pixel callout."""
        self.pixel_callout_enabled = enabled
        if not enabled:
            self.callout_label.hide()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events for pixel callout."""
        super().mouseMoveEvent(event)
        
        if not self.pixel_callout_enabled or self.original_image is None:
            return
        
        # Get mouse position in scene coordinates
        scene_pos = self.mapToScene(event.position().toPoint())
        
        # Convert to image coordinates
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        
        # Check if position is within image bounds
        if (0 <= x < self.original_image.shape[1] and 
            0 <= y < self.original_image.shape[0]):
            
            # Get pixel RGBA values
            pixel = self.original_image[y, x]
            r, g, b, a = pixel
            
            # Create callout text with better formatting
            callout_text = f"Pixel: ({x}, {y})\nR: {r:3d}  G: {g:3d}\nB: {b:3d}  A: {a:3d}"
            
            # Update callout text
            self.callout_label.setText(callout_text)
            
            # Position callout near mouse cursor in viewport coordinates
            viewport_pos = event.position().toPoint()
            callout_pos = QPointF(viewport_pos.x() + 15, viewport_pos.y() - 50)
            
            # Adjust position if callout would go outside viewport
            viewport_rect = self.viewport().rect()
            label_size = self.callout_label.sizeHint()
            
            if callout_pos.x() + label_size.width() > viewport_rect.width():
                callout_pos.setX(viewport_pos.x() - label_size.width() - 15)
            if callout_pos.y() < 0:
                callout_pos.setY(viewport_pos.y() + 15)
            
            # Set position and show callout
            self.callout_label.move(callout_pos.toPoint())
            self.callout_label.show()
        else:
            # Hide callout if outside image bounds
            self.callout_label.hide()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning while preserving callout functionality."""
        # Enable panning mode
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release to restore mouse tracking."""
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)


class TendrilVisualizationDialog(QDialog):
    """Dialog for showing tendril visualization."""
    
    def __init__(self, tendril_image, iteration, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Tendril Visualization - Iteration {iteration}")
        self.setModal(True)
        self.resize(800, 600)
        
        # Store the original image data for saving
        self.original_image = tendril_image
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"Tendril Marking - Iteration {iteration}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Image view with zoom/pan functionality and pixel callout
        self.image_view = PixelCalloutImageView(tendril_image)
        
        # Convert numpy array to QPixmap
        qimage = numpy_rgba_to_qimage(tendril_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set the pixmap using the existing scene from SynchronizedImageView
        self.image_view.set_pixmap(pixmap)
        
        layout.addWidget(self.image_view)
        
        # Pixel callout checkbox
        self.pixel_callout_checkbox = QCheckBox("Show Pixel Color Callout")
        self.pixel_callout_checkbox.toggled.connect(self.image_view.set_pixel_callout_enabled)
        layout.addWidget(self.pixel_callout_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel Algorithm")
        self.cancel_button.clicked.connect(self.reject)
        
        self.save_button = QPushButton("Save Pixels")
        self.save_button.clicked.connect(self.save_visible_pixels)
        
        self.continue_button = QPushButton("Continue Algorithm")
        self.continue_button.clicked.connect(self.accept)
        self.continue_button.setDefault(True)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.continue_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def save_visible_pixels(self):
        """Save the visible pixels from the current view to a file."""
        try:
            # Get the visible rectangle in scene coordinates
            visible_rect = self.image_view.mapToScene(self.image_view.viewport().rect()).boundingRect()
            
            # Convert to image coordinates (pixel coordinates)
            x1 = max(0, int(visible_rect.x()))
            y1 = max(0, int(visible_rect.y()))
            x2 = min(self.original_image.shape[1], int(visible_rect.x() + visible_rect.width()))
            y2 = min(self.original_image.shape[0], int(visible_rect.y() + visible_rect.height()))
            
            # Extract the visible portion
            visible_pixels = self.original_image[y1:y2, x1:x2].copy()
            
            if visible_pixels.size == 0:
                QMessageBox.warning(self, "Save Error", "No pixels visible in the current view.")
                return
            
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Visible Pixels", 
                f"tendril_iteration_{self.windowTitle().split()[-1]}_visible.npy",
                "NumPy Array (*.npy);;All Files (*)"
            )
            
            if file_path:
                # Save as numpy array
                np.save(file_path, visible_pixels)
                
                # Also save as PNG for visual reference
                png_path = file_path.replace('.npy', '.png')
                qimage = numpy_rgba_to_qimage(visible_pixels)
                qimage.save(png_path)
                
                QMessageBox.information(
                    self, 
                    "Save Successful", 
                    f"Visible pixels saved to:\n{file_path}\n\nPNG preview: {png_path}\n\nSize: {visible_pixels.shape[1]}x{visible_pixels.shape[0]} pixels"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save pixels:\n{str(e)}")


class ColorProcessingResultDialog(QDialog):
    """Dialog for showing color processing results."""
    
    def __init__(self, result_image, iteration, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Color Processing Result - Iteration {iteration}")
        self.setModal(True)
        self.resize(800, 600)
        
        # Store the original image data for saving
        self.original_image = result_image
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"Color Processing Result - Iteration {iteration}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Image view with zoom/pan functionality and pixel callout
        self.image_view = PixelCalloutImageView(result_image)
        
        # Convert numpy array to QPixmap
        qimage = numpy_rgba_to_qimage(result_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set the pixmap using the existing scene from SynchronizedImageView
        self.image_view.set_pixmap(pixmap)
        
        layout.addWidget(self.image_view)
        
        # Pixel callout checkbox
        self.pixel_callout_checkbox = QCheckBox("Show Pixel Color Callout")
        self.pixel_callout_checkbox.toggled.connect(self.image_view.set_pixel_callout_enabled)
        layout.addWidget(self.pixel_callout_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel Algorithm")
        self.cancel_button.clicked.connect(self.reject)
        
        self.save_button = QPushButton("Save Pixels")
        self.save_button.clicked.connect(self.save_visible_pixels)
        
        self.continue_button = QPushButton("Continue Algorithm")
        self.continue_button.clicked.connect(self.accept)
        self.continue_button.setDefault(True)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.continue_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def save_visible_pixels(self):
        """Save the visible pixels from the current view to a file."""
        try:
            # Get the visible rectangle in scene coordinates
            visible_rect = self.image_view.mapToScene(self.image_view.viewport().rect()).boundingRect()
            
            # Convert to image coordinates (pixel coordinates)
            x1 = max(0, int(visible_rect.x()))
            y1 = max(0, int(visible_rect.y()))
            x2 = min(self.original_image.shape[1], int(visible_rect.x() + visible_rect.width()))
            y2 = min(self.original_image.shape[0], int(visible_rect.y() + visible_rect.height()))
            
            # Extract the visible portion
            visible_pixels = self.original_image[y1:y2, x1:x2].copy()
            
            if visible_pixels.size == 0:
                QMessageBox.warning(self, "Save Error", "No pixels visible in the current view.")
                return
            
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Visible Pixels", 
                f"color_processing_iteration_{self.windowTitle().split()[-1]}_visible.npy",
                "NumPy Array (*.npy);;All Files (*)"
            )
            
            if file_path:
                # Save as numpy array
                np.save(file_path, visible_pixels)
                
                # Also save as PNG for visual reference
                png_path = file_path.replace('.npy', '.png')
                qimage = numpy_rgba_to_qimage(visible_pixels)
                qimage.save(png_path)
                
                QMessageBox.information(
                    self, 
                    "Save Successful", 
                    f"Visible pixels saved to:\n{file_path}\n\nPNG preview: {png_path}\n\nSize: {visible_pixels.shape[1]}x{visible_pixels.shape[0]} pixels"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save pixels:\n{str(e)}")


def create_tendril_visualization(rgba, threshold):
    """Create a visualization of tendrils by highlighting already-marked tendrils as magenta pixels."""
    with open('tendril_debug.log', 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] Creating tendril visualization with threshold {threshold}\n")
        f.flush()
    
    vis_image = rgba.copy()
    
    try:
        alpha = vis_image[:, :, 3]
        tendril_mask = (alpha > 10) & (alpha <= 13)
        
        tendril_count = np.sum(tendril_mask)
        
        vis_image[tendril_mask] = [255, 0, 255, 255]
        
        non_transparent = alpha > 0
        non_tendril_mask = non_transparent & ~tendril_mask
        vis_image[non_tendril_mask, 3] = 255
        
        with open('tendril_debug.log', 'a') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] Tendril visualization created: {tendril_count} magenta pixels\n")
            f.flush()
            
    except Exception as e:
        with open('tendril_debug.log', 'a') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] Error in tendril visualization: {str(e)}\n")
            f.flush()
        print(f"Error in tendril visualization: {e}")
        return rgba.copy()
    
    return vis_image

