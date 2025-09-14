#!/usr/bin/env python3
"""
Test program to compare region smoothing algorithms side by side.

This program loads a saved working image and applies each smoothing algorithm,
displaying all results in a grid for easy comparison.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QFileDialog, QScrollArea,
    QMessageBox, QGroupBox, QSlider, QCheckBox, QComboBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QSizePolicy, QTabWidget, QFormLayout, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import QPixmap, QImage, QFont, QWheelEvent, QMouseEvent, QPainter

# Add the app directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "app"))

from processing.region_cleanup import smooth_region_boundaries
from utils.qt_image import numpy_rgba_to_qimage


def qimage_to_numpy_rgba(image: QImage) -> np.ndarray:
    """Convert QImage to numpy RGBA array."""
    # Ensure 4-channel RGBA for consistent memory layout
    if image.format() != QImage.Format.Format_RGBA8888:
        img = image.convertToFormat(QImage.Format.Format_RGBA8888)
    else:
        img = image

    w = img.width()
    h = img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    return arr.copy()


class SynchronizedImageView(QGraphicsView):
    """Custom image view that can be synchronized with other views."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Disable anti-aliasing to preserve sharp region boundaries
        # self.setRenderHint(QPainter.Antialiasing)  # Commented out to prevent feathering
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create scene and pixmap item
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Synchronization state
        self._syncing = False
        self._other_views = []
        self._last_center = None
        
    def set_pixmap(self, pixmap: QPixmap):
        """Set the pixmap for this view."""
        self.pixmap_item.setPixmap(pixmap)
        self.fit_in_view()
        
    def fit_in_view(self):
        """Fit the image to the view while maintaining aspect ratio."""
        if self.pixmap_item.pixmap().isNull():
            return
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        
    def add_synchronized_view(self, view):
        """Add another view to synchronize with."""
        if view != self and view not in self._other_views:
            self._other_views.append(view)
            view._other_views.append(self)
            
    def wheelEvent(self, event: QWheelEvent):
        """Handle wheel events for zooming."""
        if event.modifiers() & Qt.ControlModifier:
            # Store the current center point before zooming
            self._last_center = self.mapToScene(self.viewport().rect().center())
            
            # Calculate zoom factor
            zoom_in_factor = 1.25
            zoom_out_factor = 1.0 / zoom_in_factor
            
            if event.angleDelta().y() > 0:
                self.scale(zoom_in_factor, zoom_in_factor)
            else:
                self.scale(zoom_out_factor, zoom_out_factor)
                
            # Synchronize with other views
            self._sync_zoom()
        else:
            super().wheelEvent(event)
            
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Store the current center for pan synchronization
            self._last_center = self.mapToScene(self.viewport().rect().center())
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        self.setDragMode(QGraphicsView.NoDrag)
        # Synchronize pan after mouse release
        if event.button() == Qt.LeftButton:
            self._sync_pan()
        super().mouseReleaseEvent(event)
        
    def _sync_zoom(self):
        """Synchronize zoom with other views."""
        if self._syncing:
            return
            
        self._syncing = True
        
        # Get current transformation matrix
        transform = self.transform()
        
        # Apply to other views
        for view in self._other_views:
            if not view._syncing:
                view._syncing = True
                # Set the same transformation
                view.setTransform(transform)
                # Center the view on the same point
                if self._last_center is not None:
                    view.centerOn(self._last_center)
                view._syncing = False
                
        self._syncing = False
        
    def _sync_pan(self):
        """Synchronize pan with other views."""
        if self._syncing:
            return
            
        self._syncing = True
        
        # Get current center point
        current_center = self.mapToScene(self.viewport().rect().center())
        
        # Apply to other views
        for view in self._other_views:
            if not view._syncing:
                view._syncing = True
                # Center the view on the same point
                view.centerOn(current_center)
                view._syncing = False
                
        self._syncing = False
        
    def sync_from_view(self, source_view):
        """Synchronize this view's transformation from another view."""
        if self._syncing:
            return
            
        self._syncing = True
        self.setTransform(source_view.transform())
        # Also sync the center point
        source_center = source_view.mapToScene(source_view.viewport().rect().center())
        self.centerOn(source_center)
        self._syncing = False


class SmoothingWorker(QThread):
    """Worker thread to apply smoothing algorithms without blocking the UI."""
    
    progress_updated = Signal(int, int, str)  # current, total, message
    algorithm_completed = Signal(str, np.ndarray, float)  # algorithm_name, result, time_taken
    
    def __init__(self, input_image, strength, preserve_colors):
        super().__init__()
        self.input_image = input_image
        self.strength = strength
        self.preserve_colors = preserve_colors
        self.algorithms = [
            "morphological",
            "bilateral", 
            "contour",
            "gaussian",
            "multiscale"
        ]
    
    def run(self):
        """Run all smoothing algorithms."""
        total = len(self.algorithms)
        
        for i, algorithm in enumerate(self.algorithms):
            self.progress_updated.emit(i + 1, total, f"Applying {algorithm} smoothing...")
            
            # Small delay to ensure progress message is visible
            time.sleep(0.1)
            
            try:
                start_time = time.time()
                result = smooth_region_boundaries(
                    self.input_image,
                    method=algorithm,
                    strength=self.strength,
                    preserve_colors=self.preserve_colors
                )
                end_time = time.time()
                time_taken = end_time - start_time
                self.algorithm_completed.emit(algorithm, result, time_taken)
            except Exception as e:
                print(f"Error applying {algorithm}: {e}")
                # Create a copy of input as fallback
                self.algorithm_completed.emit(algorithm, self.input_image.copy(), 0.0)
        
        self.progress_updated.emit(total, total, "All algorithms completed!")


class SmoothingTestWindow(QMainWindow):
    """Main window for testing smoothing algorithms."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Region Smoothing Algorithm Comparison")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.original_image = None
        self.results = {}  # algorithm_name -> (result_image, time_taken)
        self.worker = None
        self.image_views = []  # List of all image views for synchronization
        
        # Morphological tab view references
        self.morph_original_view = None
        self.morph_result_view = None
        self.morph_result_label = None
        
        # Boundary smoothing tab view references
        self.boundary_original_view = None
        self.boundary_result_view = None
        self.boundary_result_label = None
        
        # UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Tab 1: Boundary Smoothing (default)
        self.setup_boundary_smoothing_tab()
        
        # Tab 2: Morphological Fine-tuning
        self.setup_morphological_tab()
        
        # Tab 3: Algorithm Comparison
        self.setup_comparison_tab()
        
        # Progress label (shared across tabs)
        self.progress_label = QLabel("Load an image to begin comparison")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_comparison_tab(self):
        """Set up the algorithm comparison tab."""
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)
        
        # Control panel
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        # Load image button
        self.load_button = QPushButton("Load Working Image")
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)
        
        # Strength slider
        control_layout.addWidget(QLabel("Strength:"))
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self.update_strength_label)
        control_layout.addWidget(self.strength_slider)
        
        self.strength_label = QLabel("50%")
        control_layout.addWidget(self.strength_label)
        
        # Preserve colors checkbox
        self.preserve_colors_checkbox = QCheckBox("Preserve Colors")
        self.preserve_colors_checkbox.setChecked(True)
        control_layout.addWidget(self.preserve_colors_checkbox)
        
        # Run comparison button
        self.run_button = QPushButton("Run Comparison")
        self.run_button.clicked.connect(self.run_comparison)
        self.run_button.setEnabled(False)
        control_layout.addWidget(self.run_button)
        
        control_group.setLayout(control_layout)
        comparison_layout.addWidget(control_group)
        
        # Results area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QGridLayout(self.results_widget)
        self.results_layout.setSpacing(10)
        self.scroll_area.setWidget(self.results_widget)
        comparison_layout.addWidget(self.scroll_area)
        
        self.tab_widget.addTab(comparison_widget, "Algorithm Comparison")
        
    def setup_morphological_tab(self):
        """Set up the morphological fine-tuning tab."""
        morph_widget = QWidget()
        morph_layout = QVBoxLayout(morph_widget)
        
        # Control panel
        control_group = QGroupBox("Morphological Smoothing Controls")
        control_layout = QVBoxLayout(control_group)
        
        # Load image button
        load_layout = QHBoxLayout()
        self.morph_load_button = QPushButton("Load Working Image")
        self.morph_load_button.clicked.connect(self.load_image)
        load_layout.addWidget(self.morph_load_button)
        load_layout.addStretch()
        control_layout.addLayout(load_layout)
        
        # Morphological parameters
        params_group = QGroupBox("Morphological Parameters")
        params_layout = QFormLayout(params_group)
        
        # Kernel size
        self.kernel_size_spinbox = QSpinBox()
        self.kernel_size_spinbox.setRange(1, 25)
        self.kernel_size_spinbox.setValue(5)
        self.kernel_size_spinbox.setToolTip("Size of the morphological kernel (1-25 pixels)")
        params_layout.addRow("Kernel Size:", self.kernel_size_spinbox)
        
        # Kernel shape
        self.kernel_shape_combo = QComboBox()
        self.kernel_shape_combo.addItems(["Ellipse", "Rectangle", "Cross"])
        self.kernel_shape_combo.setCurrentIndex(0)
        self.kernel_shape_combo.setToolTip("Shape of the morphological kernel")
        params_layout.addRow("Kernel Shape:", self.kernel_shape_combo)
        
        # Operation sequence
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(["Open then Close", "Close then Open", "Open only", "Close only"])
        self.operation_combo.setCurrentIndex(0)
        self.operation_combo.setToolTip("Sequence of morphological operations")
        params_layout.addRow("Operation Sequence:", self.operation_combo)
        
        # Strength
        strength_layout = QHBoxLayout()
        self.morph_strength_slider = QSlider(Qt.Horizontal)
        self.morph_strength_slider.setRange(0, 100)
        self.morph_strength_slider.setValue(50)
        self.morph_strength_slider.valueChanged.connect(self.update_morph_strength_label)
        strength_layout.addWidget(self.morph_strength_slider)
        
        self.morph_strength_label = QLabel("50%")
        strength_layout.addWidget(self.morph_strength_label)
        params_layout.addRow("Strength:", strength_layout)
        
        # Preserve colors checkbox
        self.morph_preserve_colors_checkbox = QCheckBox("Preserve Colors")
        self.morph_preserve_colors_checkbox.setChecked(True)
        params_layout.addRow("", self.morph_preserve_colors_checkbox)
        
        control_layout.addWidget(params_group)
        
        # Apply button
        self.apply_morph_button = QPushButton("Apply Morphological Smoothing")
        self.apply_morph_button.clicked.connect(self.apply_morphological_smoothing)
        self.apply_morph_button.setEnabled(False)
        control_layout.addWidget(self.apply_morph_button)
        
        morph_layout.addWidget(control_group)
        
        # Results area for morphological smoothing
        self.morph_results_scroll = QScrollArea()
        self.morph_results_scroll.setWidgetResizable(True)
        self.morph_results_widget = QWidget()
        self.morph_results_layout = QHBoxLayout(self.morph_results_widget)
        self.morph_results_layout.setSpacing(10)
        self.morph_results_scroll.setWidget(self.morph_results_widget)
        morph_layout.addWidget(self.morph_results_scroll)
        
        self.tab_widget.addTab(morph_widget, "Morphological Fine-tuning")
        
    def setup_boundary_smoothing_tab(self):
        """Set up the boundary smoothing tab."""
        boundary_widget = QWidget()
        boundary_layout = QVBoxLayout(boundary_widget)
        
        # Control panel
        control_group = QGroupBox("Boundary Smoothing Controls")
        control_layout = QVBoxLayout(control_group)
        
        # Load image button
        load_layout = QHBoxLayout()
        self.boundary_load_button = QPushButton("Load Working Image")
        self.boundary_load_button.clicked.connect(self.load_image)
        load_layout.addWidget(self.boundary_load_button)
        load_layout.addStretch()
        control_layout.addLayout(load_layout)
        
        # Boundary smoothing parameters
        params_group = QGroupBox("Edge Smoothing Parameters")
        params_layout = QFormLayout(params_group)
        
        # Unlimited iterations checkbox
        self.unlimited_iterations_checkbox = QCheckBox("Unlimited Iterations")
        self.unlimited_iterations_checkbox.setToolTip("Run until no more changes occur (recommended)")
        self.unlimited_iterations_checkbox.setChecked(True)
        self.unlimited_iterations_checkbox.toggled.connect(self.on_unlimited_iterations_toggled)
        params_layout.addRow("", self.unlimited_iterations_checkbox)
        
        # Maximum iterations (disabled when unlimited is checked)
        self.gaussian_kernel_spinbox = QSpinBox()
        self.gaussian_kernel_spinbox.setRange(5, 50)
        self.gaussian_kernel_spinbox.setValue(20)
        self.gaussian_kernel_spinbox.setToolTip("Maximum iterations - only used when Unlimited Iterations is unchecked")
        self.gaussian_kernel_spinbox.setEnabled(False)
        params_layout.addRow("Max Iterations:", self.gaussian_kernel_spinbox)
        
        
        # Trim tendrils checkbox
        self.trim_tendrils_checkbox = QCheckBox("Trim Tendrils")
        self.trim_tendrils_checkbox.setToolTip("Remove thin tendrils based on thickness threshold")
        self.trim_tendrils_checkbox.setChecked(False)
        self.trim_tendrils_checkbox.toggled.connect(self.on_trim_tendrils_toggled)
        params_layout.addRow("", self.trim_tendrils_checkbox)
        
        # Minimum tendril threshold
        self.tendril_threshold_spinbox = QSpinBox()
        self.tendril_threshold_spinbox.setRange(1, 10)
        self.tendril_threshold_spinbox.setValue(2)
        self.tendril_threshold_spinbox.setToolTip("Minimum tendril width threshold (pixels)")
        self.tendril_threshold_spinbox.setEnabled(False)
        params_layout.addRow("Tendril Threshold:", self.tendril_threshold_spinbox)
        
        
        control_layout.addWidget(params_group)
        
        # Apply button
        self.apply_boundary_button = QPushButton("Apply Boundary Smoothing")
        self.apply_boundary_button.clicked.connect(self.apply_boundary_smoothing)
        self.apply_boundary_button.setEnabled(False)
        control_layout.addWidget(self.apply_boundary_button)
        
        boundary_layout.addWidget(control_group)
        
        # Results area for boundary smoothing
        self.boundary_results_scroll = QScrollArea()
        self.boundary_results_scroll.setWidgetResizable(True)
        self.boundary_results_widget = QWidget()
        self.boundary_results_layout = QHBoxLayout(self.boundary_results_widget)
        self.boundary_results_layout.setSpacing(10)
        self.boundary_results_scroll.setWidget(self.boundary_results_widget)
        boundary_layout.addWidget(self.boundary_results_scroll)
        
        self.tab_widget.addTab(boundary_widget, "Boundary Smoothing")
        
    def update_strength_label(self, value):
        """Update the strength label."""
        self.strength_label.setText(f"{value}%")
        
    def update_morph_strength_label(self, value):
        """Update the morphological strength label."""
        self.morph_strength_label.setText(f"{value}%")
        
        
    def on_unlimited_iterations_toggled(self, checked):
        """Handle unlimited iterations checkbox toggle."""
        self.gaussian_kernel_spinbox.setEnabled(not checked)
        
            
    def on_trim_tendrils_toggled(self, checked):
        """Handle trim tendrils checkbox toggle."""
        self.tendril_threshold_spinbox.setEnabled(checked)
        
    def load_image(self):
        """Load a working image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Working Image",
            "",
            "TIFF Files (*.tiff *.tif);;PNG Files (*.png);;BMP Files (*.bmp);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load the image
                qimg = QImage(file_path)
                if qimg.isNull():
                    raise ValueError("Could not load image")
                
                # Convert to numpy array
                self.original_image = qimage_to_numpy_rgba(qimg)
                
                # Reset morphological view references for new image
                self.morph_original_view = None
                self.morph_result_view = None
                self.morph_result_label = None
                
                # Reset boundary smoothing view references for new image
                self.boundary_original_view = None
                self.boundary_result_view = None
                self.boundary_result_label = None
                
                # Show original in results
                self.show_original_image()
                
                # Show original in morphological tab
                self.show_original_in_morphological_tab()
                
                # Show original in boundary smoothing tab
                self.show_original_in_boundary_tab()
                
                # Enable run buttons
                self.run_button.setEnabled(True)
                self.apply_morph_button.setEnabled(True)
                self.apply_boundary_button.setEnabled(True)
                self.progress_label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.statusBar().showMessage(f"Image loaded: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load image:\n{str(e)}")
                
    def show_original_image(self):
        """Show the original image in the results grid."""
        if self.original_image is None:
            return
            
        # Clear existing results
        self.clear_results()
        self.image_views.clear()
        
        # Convert to QPixmap for display
        qimg = numpy_rgba_to_qimage(self.original_image)
        pixmap = QPixmap.fromImage(qimg)
        
        # Create label for original
        original_label = QLabel("Original")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setFont(QFont("Arial", 10, QFont.Bold))
        original_label.setMaximumHeight(20)  # Minimize vertical space
        
        # Create synchronized image view
        original_view = SynchronizedImageView()
        original_view.set_pixmap(pixmap)
        original_view.setStyleSheet("border: 2px solid black;")
        
        # Create a container widget for the original image
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_layout.setSpacing(2)
        
        # Add label and image view to container
        original_layout.addWidget(original_label)
        original_layout.addWidget(original_view)
        
        # Add to grid (2x3 layout: original in top-left)
        self.results_layout.addWidget(original_container, 0, 0)
        
        # Store reference for synchronization
        self.image_views.append(original_view)
        
        # Set up synchronization (will be updated as more views are added)
        self._setup_synchronization()
        
    def show_original_in_morphological_tab(self):
        """Show the original image in the morphological tab."""
        if self.original_image is None:
            return
            
        # Clear previous results in morphological tab
        for i in reversed(range(self.morph_results_layout.count())):
            child = self.morph_results_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Convert to QPixmap for display
        qimg = numpy_rgba_to_qimage(self.original_image)
        pixmap = QPixmap.fromImage(qimg)
        
        # Create original image view
        self.morph_original_view = SynchronizedImageView()
        self.morph_original_view.set_pixmap(pixmap)
        
        # Create label
        original_label = QLabel("Original")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Create container
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.addWidget(original_label)
        original_layout.addWidget(self.morph_original_view)
        
        # Add to layout
        self.morph_results_layout.addWidget(original_container)
        
    def show_original_in_boundary_tab(self):
        """Show the original image in the boundary smoothing tab."""
        if self.original_image is None:
            return
            
        # Convert to QImage and display
        qimg = numpy_rgba_to_qimage(self.original_image)
        pixmap = QPixmap.fromImage(qimg)
        
        # Create original image view
        self.boundary_original_view = SynchronizedImageView()
        self.boundary_original_view.set_pixmap(pixmap)
        
        # Create original label
        original_label = QLabel("Original")
        original_label.setAlignment(Qt.AlignCenter)
        original_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        # Create original container
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.addWidget(original_label)
        original_layout.addWidget(self.boundary_original_view)
        
        # Add to layout
        self.boundary_results_layout.addWidget(original_container)
        
    def clear_results(self):
        """Clear all results from the grid."""
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def clear_algorithm_results(self):
        """Clear algorithm results but keep the original image."""
        # Clear the results dictionary
        self.results.clear()
        
        # Remove algorithm result views from image_views list (keep original)
        if len(self.image_views) > 1:
            # Keep only the first view (original image)
            original_view = self.image_views[0]
            self.image_views.clear()
            self.image_views.append(original_view)
        
        # Clear the grid layout except for the original image (position 0,0)
        # Use a more robust approach: collect widgets to remove first, then remove them
        widgets_to_remove = []
        
        # Get all widgets in the layout
        for i in range(self.results_layout.count()):
            item = self.results_layout.itemAt(i)
            if item and item.widget():
                row, col, rowspan, colspan = self.results_layout.getItemPosition(i)
                # Keep only the original image at (0,0)
                if not (row == 0 and col == 0):
                    widgets_to_remove.append(item.widget())
        
        # Remove the widgets
        for widget in widgets_to_remove:
            self.results_layout.removeWidget(widget)
            widget.deleteLater()
        
        # Force UI update to show the clearing
        QApplication.processEvents()
                
    def run_comparison(self):
        """Run the smoothing algorithm comparison."""
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        # Disable controls immediately to show something is happening
        self.run_button.setEnabled(False)
        self.load_button.setEnabled(False)
        
        # Update progress to show we're starting
        self.progress_label.setText("Starting comparison...")
        self.statusBar().showMessage("Clearing previous results and starting new comparison...")
        
        # Force UI update before clearing results
        QApplication.processEvents()
        
        # Clear previous algorithm results (but keep original image)
        self.clear_algorithm_results()
        
        # Update progress to show we're ready to process
        self.progress_label.setText("Initializing algorithms...")
        self.statusBar().showMessage("Setting up smoothing algorithms...")
        
        # Force another UI update
        QApplication.processEvents()
        
        # Get parameters
        strength = self.strength_slider.value() / 100.0
        preserve_colors = self.preserve_colors_checkbox.isChecked()
        
        # Start worker thread
        self.worker = SmoothingWorker(self.original_image, strength, preserve_colors)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.algorithm_completed.connect(self.add_result)
        self.worker.finished.connect(self.comparison_finished)
        
        self.worker.start()
        
        # Update progress to show thread has started
        self.progress_label.setText("Worker thread started, waiting for first algorithm...")
        self.statusBar().showMessage("Worker thread started...")
        
    def update_progress(self, current, total, message):
        """Update progress display."""
        progress_text = f"PROGRESS: {message} ({current}/{total})"
        self.progress_label.setText(progress_text)
        self.statusBar().showMessage(progress_text)
        
        # Force immediate UI update
        QApplication.processEvents()
        
    def add_result(self, algorithm_name, result_image, time_taken):
        """Add a result to the display grid."""
        # Convert to QPixmap for display
        qimg = numpy_rgba_to_qimage(result_image)
        pixmap = QPixmap.fromImage(qimg)
        
        # Create combined label with algorithm name and timing
        name_timing_label = QLabel(f"{algorithm_name.title()} ({time_taken:.3f}s)")
        name_timing_label.setAlignment(Qt.AlignCenter)
        name_timing_label.setFont(QFont("Arial", 10, QFont.Bold))
        name_timing_label.setMaximumHeight(20)  # Minimize vertical space
        
        # Create synchronized image view
        result_view = SynchronizedImageView()
        result_view.set_pixmap(pixmap)
        result_view.setStyleSheet("border: 2px solid black;")
        
        # Calculate position in 2x3 grid
        # Original is at (0,0)
        # Algorithms fill: (0,1), (0,2), (1,0), (1,1), (1,2)
        result_count = len(self.results)
        if result_count < 2:  # First row: columns 1, 2
            row = 0
            col = result_count + 1
        else:  # Second row: columns 0, 1, 2
            row = 1
            col = result_count - 2
        
        # Create a container widget for this algorithm result
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(2)
        
        # Add combined label and image view to container
        container_layout.addWidget(name_timing_label)
        container_layout.addWidget(result_view)
        
        # Add container to grid
        self.results_layout.addWidget(container, row, col)
        
        # Store result and add to synchronization
        self.results[algorithm_name] = (result_image, time_taken)
        self.image_views.append(result_view)
        
        # Set up synchronization with all other views
        for view in self.image_views:
            if view != result_view:
                result_view.add_synchronized_view(view)
        
        # Ensure all views are synchronized with each other
        self._setup_synchronization()
        
    def _setup_synchronization(self):
        """Set up synchronization between all image views."""
        # Clear existing synchronization
        for view in self.image_views:
            view._other_views.clear()
        
        # Set up bidirectional synchronization between all views
        for i, view1 in enumerate(self.image_views):
            for j, view2 in enumerate(self.image_views):
                if i != j:
                    view1.add_synchronized_view(view2)
        
    def comparison_finished(self):
        """Called when comparison is finished."""
        self.run_button.setEnabled(True)
        self.load_button.setEnabled(True)
        
        # Make completion message very prominent
        completion_text = f"✓ COMPLETED: All {len(self.results)} algorithms finished! Compare results above."
        self.progress_label.setText(completion_text)
        self.statusBar().showMessage("✓ All algorithms completed successfully")
        
        # Force UI update to show completion
        QApplication.processEvents()
        
        # Show completion message
        QMessageBox.information(
            self, 
            "Comparison Complete", 
            f"Applied {len(self.results)} smoothing algorithms.\n"
            "Compare the results above to choose the best algorithm."
        )
        
    def apply_morphological_smoothing(self):
        """Apply morphological smoothing with custom parameters."""
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        try:
            # Get parameters from UI
            kernel_size = self.kernel_size_spinbox.value()
            kernel_shape = self.kernel_shape_combo.currentText()
            operation = self.operation_combo.currentText()
            strength = self.morph_strength_slider.value() / 100.0
            preserve_colors = self.morph_preserve_colors_checkbox.isChecked()
            
            # Update progress
            self.progress_label.setText("Applying morphological smoothing...")
            self.statusBar().showMessage("Processing morphological smoothing...")
            QApplication.processEvents()
            
            # Apply custom morphological smoothing
            start_time = time.time()
            result = self._apply_custom_morphological_smoothing(
                self.original_image, 
                kernel_size, 
                kernel_shape, 
                operation, 
                strength, 
                preserve_colors
            )
            end_time = time.time()
            
            # Show results with parameter info
            param_info = f"K:{kernel_size} {kernel_shape[:3]} {operation[:4]} S:{int(strength*100)}%"
            self.show_morphological_results(result, end_time - start_time, param_info)
            
            # Update progress
            self.progress_label.setText(f"✓ Morphological smoothing completed in {end_time - start_time:.3f}s")
            self.statusBar().showMessage("Morphological smoothing completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"Failed to apply morphological smoothing:\n{str(e)}")
            self.progress_label.setText("Error applying morphological smoothing")
            
    def _apply_custom_morphological_smoothing(self, rgba, kernel_size, kernel_shape, operation, strength, preserve_colors):
        """Apply morphological smoothing with custom parameters."""
        import cv2 as cv
        
        result = rgba.copy()
        rgb = result[:, :, :3]
        alpha = result[:, :, 3]
        
        # Only process non-transparent pixels
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return result
        
        # Create kernel based on shape
        if kernel_shape == "Ellipse":
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == "Rectangle":
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
        else:  # Cross
            kernel = cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
        
        # Process each unique color separately to preserve palette
        unique_colors = np.unique(rgb[non_transparent].reshape(-1, 3), axis=0)
        
        # Create a temporary result to avoid overwriting during processing
        temp_result = np.zeros_like(rgba)
        
        for color in unique_colors:
            # Create mask for this color
            color_mask = np.all(rgb == color, axis=2) & non_transparent
            color_mask = color_mask.astype(np.uint8) * 255
            
            # Apply morphological operations based on sequence
            if operation == "Open then Close":
                processed = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel)
                processed = cv.morphologyEx(processed, cv.MORPH_CLOSE, kernel)
            elif operation == "Close then Open":
                processed = cv.morphologyEx(color_mask, cv.MORPH_CLOSE, kernel)
                processed = cv.morphologyEx(processed, cv.MORPH_OPEN, kernel)
            elif operation == "Open only":
                processed = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel)
            elif operation == "Close only":
                processed = cv.morphologyEx(color_mask, cv.MORPH_CLOSE, kernel)
            
            # Apply strength-based blending (always blend, even at 100% strength)
            # This ensures consistent behavior and makes the effect more visible
            processed = cv.addWeighted(color_mask, 1.0 - strength, processed, strength, 0)
            
            # Update the temporary result for this color
            temp_result[processed > 0] = [color[0], color[1], color[2], 255]
        
        # Only update pixels that were originally non-transparent
        mask = temp_result[:, :, 3] > 0
        result[mask] = temp_result[mask]
        
        return result
        
    def _apply_boundary_smoothing_global(self, rgba, gaussian_kernel, unlimited_iterations=True):
        """Apply simple 4-neighbor majority smoothing with optional gap closing."""
        import cv2 as cv
        
        result = rgba.copy()
        alpha = result[:, :, 3]
        
        # Only process non-transparent pixels
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return result
        
        # Get RGB channels
        rgb = result[:, :, :3]
        
        # Convert to a more efficient format for processing
        height, width = rgb.shape[:2]
        
        # Convert RGB to single integer representation for faster comparison
        rgb_int = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2].astype(np.uint32)
        
        # Determine max iterations
        if unlimited_iterations:
            max_iterations = 1000  # Very high limit, will stop when no changes
        else:
            max_iterations = gaussian_kernel  # Use the max iterations value directly
        
        iteration = 0
        
        while iteration < max_iterations:
            changes_made = False
            
            # Create a copy for this iteration
            new_rgb_int = rgb_int.copy()
            
            # Process each pixel (skip borders)
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # Only process non-transparent pixels
                    if not non_transparent[y, x]:
                        continue
                    
                    # Get the 4 adjacent pixels (up, down, left, right)
                    neighbors = [
                        rgb_int[y-1, x],  # up
                        rgb_int[y+1, x],  # down
                        rgb_int[y, x-1],  # left
                        rgb_int[y, x+1]   # right
                    ]
                    
                    # Get alpha values for the 4 adjacent pixels
                    neighbor_alphas = [
                        alpha[y-1, x],  # up
                        alpha[y+1, x],  # down
                        alpha[y, x-1],  # left
                        alpha[y, x+1]   # right
                    ]
                    
                    # Count occurrences of each neighbor color, but only count non-transparent neighbors
                    neighbor_counts = {}
                    for i, neighbor in enumerate(neighbors):
                        # Only count neighbors that are not transparent
                        if neighbor_alphas[i] > 0:
                            if neighbor in neighbor_counts:
                                neighbor_counts[neighbor] += 1
                            else:
                                neighbor_counts[neighbor] = 1
                    
                    # Apply smoothing rules
                    if neighbor_counts:
                        # Original majority rule: if 3+ neighbors have the same color
                        most_common_color = max(neighbor_counts, key=neighbor_counts.get)
                        most_common_count = neighbor_counts[most_common_color]
                        
                        if most_common_count >= 3:
                            new_rgb_int[y, x] = most_common_color
                            changes_made = True
            
            # Update the RGB array
            rgb_int = new_rgb_int
            
            # Convert back to RGB format
            result[:, :, 0] = (rgb_int >> 16) & 0xFF
            result[:, :, 1] = (rgb_int >> 8) & 0xFF
            result[:, :, 2] = rgb_int & 0xFF
            
            iteration += 1
            
            # If no changes were made, we're done
            if not changes_made:
                break
        
        return result
        
    def _apply_boundary_smoothing_global_with_progress(self, rgba, gaussian_kernel, unlimited_iterations=True, trim_tendrils=False, tendril_threshold=2):
        """Apply boundary smoothing with progress tracking and convergence detection."""
        import cv2 as cv
        
        result = rgba.copy()
        alpha = result[:, :, 3]
        
        # Only process non-transparent pixels
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return result
        
        # Get RGB channels
        rgb = result[:, :, :3]
        
        # Convert to a more efficient format for processing
        height, width = rgb.shape[:2]
        
        # Convert RGB to single integer representation for faster comparison
        rgb_int = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2].astype(np.uint32)
        
        # Determine max iterations
        if unlimited_iterations:
            max_iterations = 1000  # Very high limit, will stop when no changes
        else:
            max_iterations = gaussian_kernel  # Use the max iterations value directly
        
        iteration = 0
        total_pixels = np.sum(non_transparent)
        changes_history = []  # Track changes over time for convergence detection
        start_time = time.time()
        
        while iteration < max_iterations:
            changes_made = False
            pixels_changed = 0
            
            
            # Create a copy for this iteration
            new_rgb_int = rgb_int.copy()
            
            # Process each pixel (skip borders)
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # Only process non-transparent pixels
                    if not non_transparent[y, x]:
                        continue
                    
                    # Get the 4 adjacent pixels (up, down, left, right)
                    neighbors = [
                        rgb_int[y-1, x],  # up
                        rgb_int[y+1, x],  # down
                        rgb_int[y, x-1],  # left
                        rgb_int[y, x+1]   # right
                    ]
                    
                    # Get alpha values for the 4 adjacent pixels
                    neighbor_alphas = [
                        alpha[y-1, x],  # up
                        alpha[y+1, x],  # down
                        alpha[y, x-1],  # left
                        alpha[y, x+1]   # right
                    ]
                    
                    # Count occurrences of each neighbor color, but only count non-transparent neighbors
                    neighbor_counts = {}
                    for i, neighbor in enumerate(neighbors):
                        # Only count neighbors that are not transparent
                        if neighbor_alphas[i] > 0:
                            if neighbor in neighbor_counts:
                                neighbor_counts[neighbor] += 1
                            else:
                                neighbor_counts[neighbor] = 1
                    
                    # Apply smoothing rules
                    if neighbor_counts:
                        # Original majority rule: if 3+ neighbors have the same color
                        most_common_color = max(neighbor_counts, key=neighbor_counts.get)
                        most_common_count = neighbor_counts[most_common_color]
                        
                        if most_common_count >= 3:
                            new_rgb_int[y, x] = most_common_color
                            changes_made = True
                            pixels_changed += 1
                
            
            # Update the RGB array
            rgb_int = new_rgb_int
            
            # Convert back to RGB format
            result[:, :, 0] = (rgb_int >> 16) & 0xFF
            result[:, :, 1] = (rgb_int >> 8) & 0xFF
            result[:, :, 2] = rgb_int & 0xFF
            
            # Apply tendril trimming AFTER RGB conversion (so it doesn't get overwritten)
            if trim_tendrils:
                from processing.tendril_trimming import TendrilTrimmer
                trimmer = TendrilTrimmer()
                tendrils_removed = trimmer._trim_tendrils_in_iteration(result, tendril_threshold)
                if tendrils_removed > 0:
                    changes_made = True
                    pixels_changed += tendrils_removed
                    
                    # Apply color selection to magenta pixels
                    pixels_recolored = trimmer._apply_color_selection_to_magenta(result)
                    if pixels_recolored > 0:
                        pixels_changed += pixels_recolored
                    
                    # Continue iterating to handle more tendrils
                    # Don't break - let the main loop continue
            
            iteration += 1
            
            # Track changes for convergence detection
            change_percentage = (pixels_changed / total_pixels) * 100 if total_pixels > 0 else 0
            changes_history.append(change_percentage)
            
            # Update progress
            elapsed_time = time.time() - start_time
            self.progress_label.setText(f"Iteration {iteration}: {pixels_changed} pixels changed ({change_percentage:.1f}%) - {elapsed_time:.1f}s")
            QApplication.processEvents()
            
            # Convergence detection - be more conservative
            if not changes_made:
                # No changes made - converged
                self.progress_label.setText(f"✓ Converged after {iteration} iterations in {elapsed_time:.1f}s")
                break
            
            # Check for non-convergence patterns - only after more iterations
            if len(changes_history) >= 20:
                # Check if we're in a repeating pattern (oscillating)
                recent_changes = changes_history[-20:]
                if len(set([round(x, 1) for x in recent_changes])) <= 2:
                    # Very similar change rates - might be oscillating
                    self.progress_label.setText(f"⚠ Stopped after {iteration} iterations - possible oscillation detected")
                    break
            
            # Safety timeout (30 seconds for tendril trimming, 30 for regular)
            timeout_limit = 30 if trim_tendrils else 30
            if elapsed_time > timeout_limit:
                self.progress_label.setText(f"⚠ Stopped after {iteration} iterations - timeout reached")
                break
            
            # Additional safety: if tendril trimming is removing too many pixels, it might be in a loop
            if trim_tendrils and pixels_changed > total_pixels * 0.5:
                self.progress_label.setText(f"⚠ Stopped after {iteration} iterations - too many pixels changed (possible loop)")
                break
            
            # Keep only last 20 change records to prevent memory growth
            if len(changes_history) > 20:
                changes_history = changes_history[-20:]
        
        # Apply tendril trimming using the extracted algorithm
        if trim_tendrils:
            from processing.tendril_trimming import trim_tendrils as trim_tendrils_algorithm
            
            # Use the extracted algorithm
            result, iterations_used, status_message = trim_tendrils_algorithm(result, tendril_threshold, max_iterations=30)
            
            if self.progress_label:
                self.progress_label.setText(f"Tendril cleanup: {status_message}")
        
        return result
        
        
    def show_morphological_results(self, result_image, time_taken, param_info=""):
        """Show morphological smoothing results while preserving zoom/pan state."""
        # If this is the first time, create the views
        if self.morph_result_view is None:
            # Create result image view
            result_qimg = numpy_rgba_to_qimage(result_image)
            result_pixmap = QPixmap.fromImage(result_qimg)
            self.morph_result_view = SynchronizedImageView()
            self.morph_result_view.set_pixmap(result_pixmap)
            
            # Synchronize views
            if self.morph_original_view is not None:
                self.morph_original_view.add_synchronized_view(self.morph_result_view)
                self.morph_result_view.add_synchronized_view(self.morph_original_view)
            
            # Create result label
            self.morph_result_label = QLabel(f"Result ({time_taken:.3f}s)\n{param_info}")
            self.morph_result_label.setAlignment(Qt.AlignCenter)
            self.morph_result_label.setFont(QFont("Arial", 10, QFont.Bold))
            
            # Create result container
            result_container = QWidget()
            result_layout = QVBoxLayout(result_container)
            result_layout.addWidget(self.morph_result_label)
            result_layout.addWidget(self.morph_result_view)
            
            # Add to layout
            self.morph_results_layout.addWidget(result_container)
        else:
            # Update existing result view with new image while preserving zoom/pan
            result_qimg = numpy_rgba_to_qimage(result_image)
            result_pixmap = QPixmap.fromImage(result_qimg)
            
            # Store current view state using scroll bar positions for pixel-perfect precision
            current_transform = self.morph_result_view.transform()
            h_scrollbar = self.morph_result_view.horizontalScrollBar()
            v_scrollbar = self.morph_result_view.verticalScrollBar()
            h_value = h_scrollbar.value()
            v_value = v_scrollbar.value()
            
            # Temporarily disable fit_in_view to prevent automatic adjustment
            original_fit_in_view = self.morph_result_view.fit_in_view
            self.morph_result_view.fit_in_view = lambda: None  # Disable temporarily
            
            # Update the pixmap directly without calling fit_in_view
            self.morph_result_view.pixmap_item.setPixmap(result_pixmap)
            
            # Restore the fit_in_view method
            self.morph_result_view.fit_in_view = original_fit_in_view
            
            # Restore exact transformation and scroll positions
            self.morph_result_view.setTransform(current_transform)
            h_scrollbar.setValue(h_value)
            v_scrollbar.setValue(v_value)
            
            # Update the label with new timing and parameter info
            if self.morph_result_label is not None:
                self.morph_result_label.setText(f"Result ({time_taken:.3f}s)\n{param_info}")
                
    def apply_boundary_smoothing(self):
        """Apply boundary smoothing with custom parameters."""
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        try:
            # Get parameters from UI
            gaussian_kernel = self.gaussian_kernel_spinbox.value()
            unlimited_iterations = self.unlimited_iterations_checkbox.isChecked()
            trim_tendrils = self.trim_tendrils_checkbox.isChecked()
            tendril_threshold = self.tendril_threshold_spinbox.value()
            
            # Debug: Show what parameters are being used
            self.progress_label.setText(f"Parameters: unlimited={unlimited_iterations}, trim_tendrils={trim_tendrils}, threshold={tendril_threshold}")
            QApplication.processEvents()
            
            # Update progress
            self.progress_label.setText("Applying boundary smoothing...")
            self.statusBar().showMessage("Processing boundary smoothing...")
            QApplication.processEvents()
            
            
            # Apply boundary smoothing with progress tracking
            start_time = time.time()
            result = self._apply_boundary_smoothing_global_with_progress(
                self.original_image, 
                gaussian_kernel, unlimited_iterations, trim_tendrils, tendril_threshold
            )
            end_time = time.time()
            
            # Show results with parameter info
            param_parts = ["4-Neighbor"]
            if trim_tendrils:
                param_parts.append(f"TEND:{tendril_threshold}")
            if not unlimited_iterations:
                param_parts.append(f"MAX:{gaussian_kernel}")
            
            param_info = " ".join(param_parts)
            self.show_boundary_results(result, end_time - start_time, param_info)
            
            # Update progress
            self.progress_label.setText(f"✓ Boundary smoothing completed in {end_time - start_time:.3f}s")
            self.statusBar().showMessage("Boundary smoothing completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"Failed to apply boundary smoothing:\n{str(e)}")
            self.progress_label.setText("Error applying boundary smoothing")
            
    def show_boundary_results(self, result_image, time_taken, param_info=""):
        """Show boundary smoothing results while preserving zoom/pan state."""
        # If this is the first time, create the views
        if self.boundary_result_view is None:
            # Create result image view
            result_qimg = numpy_rgba_to_qimage(result_image)
            result_pixmap = QPixmap.fromImage(result_qimg)
            self.boundary_result_view = SynchronizedImageView()
            self.boundary_result_view.set_pixmap(result_pixmap)
            
            # Synchronize views
            if self.boundary_original_view is not None:
                self.boundary_original_view.add_synchronized_view(self.boundary_result_view)
                self.boundary_result_view.add_synchronized_view(self.boundary_original_view)
            
            # Create result label
            self.boundary_result_label = QLabel(f"Result ({time_taken:.3f}s)\n{param_info}")
            self.boundary_result_label.setAlignment(Qt.AlignCenter)
            self.boundary_result_label.setFont(QFont("Arial", 10, QFont.Bold))
            
            # Create result container
            result_container = QWidget()
            result_layout = QVBoxLayout(result_container)
            result_layout.addWidget(self.boundary_result_label)
            result_layout.addWidget(self.boundary_result_view)
            
            # Add to layout
            self.boundary_results_layout.addWidget(result_container)
        else:
            # Update existing result view with new image while preserving zoom/pan
            result_qimg = numpy_rgba_to_qimage(result_image)
            result_pixmap = QPixmap.fromImage(result_qimg)
            
            # Store current view state using scroll bar positions for pixel-perfect precision
            current_transform = self.boundary_result_view.transform()
            h_scrollbar = self.boundary_result_view.horizontalScrollBar()
            v_scrollbar = self.boundary_result_view.verticalScrollBar()
            h_value = h_scrollbar.value()
            v_value = v_scrollbar.value()
            
            # Temporarily disable fit_in_view to prevent automatic adjustment
            original_fit_in_view = self.boundary_result_view.fit_in_view
            self.boundary_result_view.fit_in_view = lambda: None  # Disable temporarily
            
            # Update the pixmap directly without calling fit_in_view
            self.boundary_result_view.pixmap_item.setPixmap(result_pixmap)
            
            # Restore the fit_in_view method
            self.boundary_result_view.fit_in_view = original_fit_in_view
            
            # Restore exact transformation and scroll positions
            self.boundary_result_view.setTransform(current_transform)
            h_scrollbar.setValue(h_value)
            v_scrollbar.setValue(v_value)
            
            # Update the label with new timing and parameter info
            if self.boundary_result_label is not None:
                self.boundary_result_label.setText(f"Result ({time_taken:.3f}s)\n{param_info}")


def main():
    """Main function."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Smoothing Algorithm Tester")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = SmoothingTestWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
