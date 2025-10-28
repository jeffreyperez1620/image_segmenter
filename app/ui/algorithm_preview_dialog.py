from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QPoint, QRect, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent, QCursor, QTransform
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGridLayout, QScrollArea, QWidget, QFrame, QProgressBar,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSizePolicy
)

from processing.color_simplify import (
    simplify_colors_kmeans, simplify_colors_median_cut, simplify_colors_octree,
    simplify_colors_threshold, simplify_colors_perceptual, simplify_colors_perceptual_fast,
    simplify_colors_adaptive_distance, simplify_colors_hsv_clustering
)
from utils.qt_image import numpy_rgba_to_qimage


class SynchronizedImageView(QGraphicsView):
    """A graphics view that can be synchronized with other views for pan/zoom."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Set transparent background
        self.setStyleSheet("background-color: transparent;")
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        
        # Synchronization
        self.synchronized_views = []
        self.is_syncing = False
        
        # Scene setup
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(Qt.transparent)
        self.setScene(self.scene)
        
        # Image item
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        
        # Mouse tracking for panning
        self.last_pan_point = QPointF()
        self.panning = False
    
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

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for real-time panning sync."""
        if self.panning and event.buttons() == Qt.LeftButton:
            # Sync pan in real-time during dragging
            self._sync_pan()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.panning = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Final sync after release
            self._sync_pan()
        super().mouseReleaseEvent(event)
    
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
                # Also sync scroll positions to ensure perfect alignment
                view.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
                view.verticalScrollBar().setValue(self.verticalScrollBar().value())
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
                # Also sync transform to ensure perfect alignment
                view.setTransform(self.transform())
                view.is_syncing = False

    def sync_from_view(self, source_view):
        """Synchronize this view with a source view."""
        if self.is_syncing:
            return
            
        self.is_syncing = True
        
        # Copy transform first
        self.setTransform(source_view.transform())
        
        # Copy scroll positions
        self.horizontalScrollBar().setValue(source_view.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(source_view.verticalScrollBar().value())
        
        # Force update to ensure perfect alignment
        self.update()
        
        self.is_syncing = False


class AlgorithmPreviewWorker(QThread):
    """Worker thread for processing algorithms in the background."""
    
    progress_updated = Signal(int, str)  # progress, algorithm_name
    algorithm_completed = Signal(str, np.ndarray, np.ndarray, float)  # algorithm, result, palette, time
    all_completed = Signal()
    
    def __init__(self, rgba: np.ndarray, target_colors: int):
        super().__init__()
        self.rgba = rgba
        self.target_colors = target_colors
        
        # Define algorithms to process (excluding auto-select)
        self.algorithms = [
            ("K-means Clustering", "kmeans"),
            ("Median Cut", "median_cut"),
            ("Threshold/Posterize", "threshold"),
            ("Perceptual Clustering", "perceptual"),
            ("Perceptual Clustering (Fast)", "perceptual_fast"),
            ("HSV Clustering", "hsv_clustering"),
        ]
    
    def run(self):
        """Process all algorithms and emit results."""
        total_algorithms = len(self.algorithms)
        
        for i, (display_name, algorithm_key) in enumerate(self.algorithms):
            try:
                # Update progress
                self.progress_updated.emit(i, display_name)
                
                # Process algorithm
                start_time = time.time()
                result_rgba, palette = self._process_algorithm(algorithm_key)
                processing_time = time.time() - start_time
                
                # Emit result
                self.algorithm_completed.emit(display_name, result_rgba, palette, processing_time)
                
            except Exception as e:
                print(f"Error processing {display_name}: {e}")
                # Continue with next algorithm
                continue
        
        # Signal completion
        self.all_completed.emit()
    
    def _process_algorithm(self, algorithm_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single algorithm."""
        if algorithm_key == "kmeans":
            return simplify_colors_kmeans(self.rgba, self.target_colors)
        elif algorithm_key == "median_cut":
            return simplify_colors_median_cut(self.rgba, self.target_colors)
        elif algorithm_key == "threshold":
            return simplify_colors_threshold(self.rgba, self.target_colors)
        elif algorithm_key == "perceptual":
            return simplify_colors_perceptual(self.rgba, self.target_colors)
        elif algorithm_key == "perceptual_fast":
            return simplify_colors_perceptual_fast(self.rgba, self.target_colors)
        elif algorithm_key == "hsv_clustering":
            return simplify_colors_hsv_clustering(self.rgba, self.target_colors)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_key}")


class AlgorithmPreviewCell(QFrame):
    """A cell in the algorithm preview grid."""
    
    algorithm_selected = Signal(str, np.ndarray, np.ndarray)  # algorithm_name, result, palette
    
    def __init__(self, algorithm_name: str, parent=None):
        super().__init__(parent)
        self.algorithm_name = algorithm_name
        self.result_rgba = None
        self.palette = None
        self.processing_time = 0.0
        
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)
        self.setMinimumSize(200, 200)  # Minimum size, but can stretch
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the cell UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Algorithm name (fixed size)
        self.name_label = QLabel(self.algorithm_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.name_label.setFixedHeight(20)  # Fixed height
        layout.addWidget(self.name_label)
        
        # Processing time (fixed size)
        self.time_label = QLabel("Processing...")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 10px; color: #666;")
        self.time_label.setFixedHeight(15)  # Fixed height
        layout.addWidget(self.time_label)
        
        # Image preview with pan/zoom (stretches)
        self.image_view = SynchronizedImageView()
        self.image_view.setMinimumSize(150, 100)  # Minimum size
        self.image_view.setStyleSheet("border: 1px solid #ccc; background-color: transparent;")
        layout.addWidget(self.image_view, 1)  # Stretch factor = 1
        
        # Select button (fixed size)
        self.select_button = QPushButton("Select")
        self.select_button.setEnabled(False)
        self.select_button.clicked.connect(self._on_select)
        self.select_button.setFixedHeight(30)  # Fixed height
        layout.addWidget(self.select_button)
    
    def update_result(self, result_rgba: np.ndarray, palette: np.ndarray, processing_time: float):
        """Update the cell with algorithm results."""
        self.result_rgba = result_rgba
        self.palette = palette
        self.processing_time = processing_time
        
        # Update time label
        self.time_label.setText(f"Time: {processing_time:.3f}s")
        
        # Convert result to QPixmap and display
        qimage = numpy_rgba_to_qimage(result_rgba)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set pixmap in the graphics view
        self.image_view.set_pixmap(pixmap)
        
        # Enable select button
        self.select_button.setEnabled(True)
    
    def _on_select(self):
        """Handle select button click."""
        if self.result_rgba is not None and self.palette is not None:
            self.algorithm_selected.emit(self.algorithm_name, self.result_rgba, self.palette)


class AlgorithmPreviewDialog(QDialog):
    """Dialog showing preview of all color processing algorithms."""
    
    algorithm_selected = Signal(str, np.ndarray, np.ndarray)  # algorithm_name, result, palette
    
    def __init__(self, rgba: np.ndarray, target_colors: int, parent=None):
        super().__init__(parent)
        self.rgba = rgba
        self.target_colors = target_colors
        self.cells = {}
        self.synchronized_views = []  # List of all image views for synchronization
        
        self.setWindowTitle("Preview All Color Processing Algorithms")
        self.setModal(True)
        self.resize(1200, 800)
        
        self._init_ui()
        self._start_processing()
    
    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Color Processing Algorithm Results")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Scroll area for grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Grid widget
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(5)  # Reduced padding between cells
        
        # Set stretch factors for columns to make grid stretch
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setColumnStretch(2, 1)
        
        scroll_area.setWidget(self.grid_widget)
        layout.addWidget(scroll_area)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        layout.addWidget(close_button)
    
    def _start_processing(self):
        """Start processing all algorithms."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 6)  # 6 algorithms
        
        # Create worker thread
        self.worker = AlgorithmPreviewWorker(self.rgba, self.target_colors)
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.algorithm_completed.connect(self._on_algorithm_completed)
        self.worker.all_completed.connect(self._on_all_completed)
        self.worker.start()
    
    def _on_progress_updated(self, progress: int, algorithm_name: str):
        """Handle progress updates."""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"Processing {algorithm_name}...")
    
    def _on_algorithm_completed(self, algorithm_name: str, result_rgba: np.ndarray, palette: np.ndarray, processing_time: float):
        """Handle algorithm completion."""
        # Create cell for this algorithm
        cell = AlgorithmPreviewCell(algorithm_name)
        cell.algorithm_selected.connect(self._on_algorithm_selected)
        cell.update_result(result_rgba, palette, processing_time)
        
        # Add to grid (3 columns for 6 algorithms = 2 rows)
        row = len(self.cells) // 3  # 3 columns
        col = len(self.cells) % 3
        self.grid_layout.addWidget(cell, row, col)
        
        # Add to synchronized views list
        self.synchronized_views.append(cell.image_view)
        
        # Connect synchronization - add this view to all existing views
        for existing_view in self.synchronized_views[:-1]:
            existing_view.add_synchronized_view(cell.image_view)
            cell.image_view.add_synchronized_view(existing_view)
        
        # If this is not the first view, sync it with the first one
        if len(self.synchronized_views) > 1:
            first_view = self.synchronized_views[0]
            cell.image_view.sync_from_view(first_view)
        
        self.cells[algorithm_name] = cell
    
    def _on_all_completed(self):
        """Handle completion of all algorithms."""
        self.progress_bar.setVisible(False)
        
        # Ensure all views are synchronized after all algorithms complete
        if len(self.synchronized_views) > 1:
            # Use the first view as the reference
            reference_view = self.synchronized_views[0]
            for view in self.synchronized_views[1:]:
                view.sync_from_view(reference_view)
            
            # Force a final sync to ensure perfect alignment
            self._force_sync_all_views()
    
    def _force_sync_all_views(self):
        """Force all views to sync with the first view for perfect alignment."""
        if len(self.synchronized_views) < 2:
            return
            
        reference_view = self.synchronized_views[0]
        
        # Get reference transform and scroll positions
        ref_transform = reference_view.transform()
        ref_h_scroll = reference_view.horizontalScrollBar().value()
        ref_v_scroll = reference_view.verticalScrollBar().value()
        
        # Apply to all other views
        for view in self.synchronized_views[1:]:
            view.is_syncing = True
            view.setTransform(ref_transform)
            view.horizontalScrollBar().setValue(ref_h_scroll)
            view.verticalScrollBar().setValue(ref_v_scroll)
            view.update()
            view.is_syncing = False
    
    def _on_algorithm_selected(self, algorithm_name: str, result_rgba: np.ndarray, palette: np.ndarray):
        """Handle algorithm selection."""
        self.algorithm_selected.emit(algorithm_name, result_rgba, palette)
        self.accept()  # Close dialog
