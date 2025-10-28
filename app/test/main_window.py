#!/usr/bin/env python3
"""
Main window for the smoothing test application.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QFileDialog, QScrollArea,
    QMessageBox, QGroupBox, QSlider, QCheckBox, QComboBox, 
    QSizePolicy, QTabWidget, QFormLayout, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import QPixmap, QImage, QFont, QWheelEvent, QMouseEvent, QPainter

# Add the app directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.qt_image import numpy_rgba_to_qimage
from synchronized_image_view import SynchronizedImageView
from smoothing_worker import SmoothingWorker
from tendril_algorithm import TendrilAlgorithm
from visualization import TendrilVisualizationDialog, ColorProcessingResultDialog, create_tendril_visualization
from test_utils import qimage_to_numpy_rgba


class SmoothingTestWindow(QMainWindow):
    """Main window for the smoothing test application."""
    
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.boundary_result_view = None
        self.morph_result_view = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Smoothing Algorithm Test")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.setup_boundary_smoothing_tab()  # Make tendril trimming first
        self.setup_comparison_tab()
        self.setup_morphological_tab()
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        central_widget.setLayout(layout)
        
    def setup_comparison_tab(self):
        """Set up the comparison tab."""
        comparison_widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        
        self.run_button = QPushButton("Run Comparison")
        self.run_button.clicked.connect(self.run_comparison)
        self.run_button.setEnabled(False)
        
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear_results)
        
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Progress
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)
        
        # Results grid
        self.results_scroll = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QGridLayout()
        self.results_widget.setLayout(self.results_layout)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setWidgetResizable(True)
        
        layout.addWidget(self.results_scroll)
        
        comparison_widget.setLayout(layout)
        self.tab_widget.addTab(comparison_widget, "Comparison")
        
    def setup_morphological_tab(self):
        """Set up the morphological smoothing tab."""
        morph_widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.morph_load_button = QPushButton("Load Image")
        self.morph_load_button.clicked.connect(self.load_image)
        
        self.morph_show_original_button = QPushButton("Show Original")
        self.morph_show_original_button.clicked.connect(self.show_original_in_morphological_tab)
        self.morph_show_original_button.setEnabled(False)
        
        self.morph_apply_button = QPushButton("Apply Morphological Smoothing")
        self.morph_apply_button.clicked.connect(self.apply_morphological_smoothing)
        self.morph_apply_button.setEnabled(False)
        
        self.morph_clear_button = QPushButton("Clear Results")
        self.morph_clear_button.clicked.connect(self.clear_algorithm_results)
        
        controls_layout.addWidget(self.morph_load_button)
        controls_layout.addWidget(self.morph_show_original_button)
        controls_layout.addWidget(self.morph_apply_button)
        controls_layout.addWidget(self.morph_clear_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        self.morph_kernel_size_spinbox = QSpinBox()
        self.morph_kernel_size_spinbox.setRange(3, 21)
        self.morph_kernel_size_spinbox.setValue(5)
        self.morph_kernel_size_spinbox.setSingleStep(2)
        params_layout.addRow("Kernel Size:", self.morph_kernel_size_spinbox)
        
        self.morph_kernel_shape_combo = QComboBox()
        self.morph_kernel_shape_combo.addItems(["Rectangle", "Ellipse", "Cross"])
        params_layout.addRow("Kernel Shape:", self.morph_kernel_shape_combo)
        
        self.morph_operation_combo = QComboBox()
        self.morph_operation_combo.addItems(["Opening", "Closing", "Gradient", "Top Hat", "Black Hat"])
        params_layout.addRow("Operation:", self.morph_operation_combo)
        
        self.morph_strength_slider = QSlider(Qt.Horizontal)
        self.morph_strength_slider.setRange(1, 10)
        self.morph_strength_slider.setValue(5)
        self.morph_strength_slider.valueChanged.connect(self.update_morph_strength_label)
        params_layout.addRow("Strength:", self.morph_strength_slider)
        
        self.morph_strength_label = QLabel("5")
        params_layout.addRow("", self.morph_strength_label)
        
        self.morph_preserve_colors_checkbox = QCheckBox("Preserve Colors")
        self.morph_preserve_colors_checkbox.setChecked(True)
        params_layout.addRow("", self.morph_preserve_colors_checkbox)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Results
        self.morph_result_view = SynchronizedImageView()
        layout.addWidget(self.morph_result_view)
        
        morph_widget.setLayout(layout)
        self.tab_widget.addTab(morph_widget, "Morphological")
        
    def setup_boundary_smoothing_tab(self):
        """Set up the boundary smoothing tab."""
        boundary_widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.boundary_load_button = QPushButton("Load Image")
        self.boundary_load_button.clicked.connect(self.load_image)
        
        self.boundary_apply_button = QPushButton("Apply Tendril Trimming")
        self.boundary_apply_button.clicked.connect(self.apply_boundary_smoothing)
        self.boundary_apply_button.setEnabled(False)
        
        self.boundary_clear_button = QPushButton("Clear Results")
        self.boundary_clear_button.clicked.connect(self.clear_algorithm_results)
        
        controls_layout.addWidget(self.boundary_load_button)
        controls_layout.addWidget(self.boundary_apply_button)
        controls_layout.addWidget(self.boundary_clear_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        self.gaussian_kernel_spinbox = QSpinBox()
        self.gaussian_kernel_spinbox.setRange(3, 21)
        self.gaussian_kernel_spinbox.setValue(5)
        self.gaussian_kernel_spinbox.setSingleStep(2)
        params_layout.addRow("Gaussian Kernel:", self.gaussian_kernel_spinbox)
        
        self.unlimited_iterations_checkbox = QCheckBox("Unlimited Iterations")
        self.unlimited_iterations_checkbox.setChecked(True)
        self.unlimited_iterations_checkbox.toggled.connect(self.on_unlimited_iterations_toggled)
        params_layout.addRow("", self.unlimited_iterations_checkbox)
        
        self.tendril_threshold_spinbox = QSpinBox()
        self.tendril_threshold_spinbox.setRange(1, 10)
        self.tendril_threshold_spinbox.setValue(2)
        params_layout.addRow("Tendril Threshold:", self.tendril_threshold_spinbox)
        
        self.step_mode_checkbox = QCheckBox("Step Mode (Interactive)")
        self.step_mode_checkbox.setChecked(False)
        params_layout.addRow("", self.step_mode_checkbox)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Results
        self.boundary_result_view = SynchronizedImageView()
        layout.addWidget(self.boundary_result_view)
        
        boundary_widget.setLayout(layout)
        self.tab_widget.addTab(boundary_widget, "Tendril Trimming")
        
    def update_strength_label(self, value):
        """Update the strength label."""
        self.strength_label.setText(str(value))
        
    def update_morph_strength_label(self, value):
        """Update the morphological strength label."""
        self.morph_strength_label.setText(str(value))
        
    def on_unlimited_iterations_toggled(self, checked):
        """Handle unlimited iterations checkbox toggle."""
        # This method can be used to enable/disable related controls
        pass
        
    def load_image(self):
        """Load an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            try:
                # Load image using QImage
                qimage = QImage(file_path)
                if qimage.isNull():
                    QMessageBox.critical(self, "Load Error", "Failed to load image file.")
                    return
                
                # Convert to numpy array
                self.original_image = qimage_to_numpy_rgba(qimage)
                
                # Enable buttons
                self.run_button.setEnabled(True)
                self.morph_show_original_button.setEnabled(True)
                self.morph_apply_button.setEnabled(True)
                self.boundary_apply_button.setEnabled(True)
                
                # Automatically show original image in tendril trimming tab
                self.show_original_in_boundary_tab()
                
                self.progress_label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.statusBar().showMessage(f"Image loaded: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load image:\n{str(e)}")
                
    def show_original_image(self):
        """Show the original image in the results grid."""
        if self.original_image is None:
            return
            
        # Convert to QPixmap
        qimage = numpy_rgba_to_qimage(self.original_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Create view
        view = SynchronizedImageView()
        view.set_pixmap(pixmap)
        
        # Add to grid
        self.results_layout.addWidget(QLabel("Original"), 0, 0)
        self.results_layout.addWidget(view, 1, 0)
        
        # Set up synchronization
        self._setup_synchronization()
        
    def show_original_in_morphological_tab(self):
        """Show the original image in the morphological tab."""
        if self.original_image is None:
            return
            
        # Convert to QPixmap
        qimage = numpy_rgba_to_qimage(self.original_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set in view
        self.morph_result_view.set_pixmap(pixmap)
        
    def show_original_in_boundary_tab(self):
        """Show the original image in the boundary tab."""
        if self.original_image is None:
            return
            
        # Convert to QPixmap
        qimage = numpy_rgba_to_qimage(self.original_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set in view
        self.boundary_result_view.set_pixmap(pixmap)
        
    def clear_results(self):
        """Clear all results from the comparison tab."""
        # Clear the grid layout
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def clear_algorithm_results(self):
        """Clear results from algorithm tabs."""
        if self.morph_result_view:
            self.morph_result_view.scene.clear()
        if self.boundary_result_view:
            self.boundary_result_view.scene.clear()
            
    def run_comparison(self):
        """Run the comparison of smoothing algorithms."""
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        # Clear previous results
        self.clear_results()
        
        # Show original
        self.show_original_image()
        
        # Get parameters
        strength = self.strength_slider.value()
        preserve_colors = self.preserve_colors_checkbox.isChecked()
        
        # Create and start worker
        self.worker = SmoothingWorker(self.original_image, strength, preserve_colors)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.algorithm_completed.connect(self.add_result)
        self.worker.finished.connect(self.comparison_finished)
        self.worker.start()
        
    def update_progress(self, current, total, message):
        """Update progress display."""
        self.progress_label.setText(f"{message} ({current}/{total})")
        
    def add_result(self, algorithm_name, result_image, time_taken):
        """Add a result to the comparison grid."""
        # Convert to QPixmap
        qimage = numpy_rgba_to_qimage(result_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Create view
        view = SynchronizedImageView()
        view.set_pixmap(pixmap)
        
        # Find next available position
        row = self.results_layout.rowCount()
        
        # Add label and view
        self.results_layout.addWidget(QLabel(algorithm_name), row, 0)
        self.results_layout.addWidget(view, row + 1, 0)
        
        # Set up synchronization
        self._setup_synchronization()
        
    def _setup_synchronization(self):
        """Set up synchronization between all image views."""
        views = []
        for i in range(self.results_layout.count()):
            widget = self.results_layout.itemAt(i).widget()
            if isinstance(widget, SynchronizedImageView):
                views.append(widget)
        
        # Set up synchronization
        for view in views:
            for other_view in views:
                if view != other_view:
                    view.add_synchronized_view(other_view)
                    
    def comparison_finished(self):
        """Handle comparison completion."""
        self.progress_label.setText("Comparison completed!")
        self.statusBar().showMessage("Comparison completed")
        
    def apply_morphological_smoothing(self):
        """Apply morphological smoothing."""
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        try:
            # Get parameters
            kernel_size = self.morph_kernel_size_spinbox.value()
            kernel_shape = self.morph_kernel_shape_combo.currentText().lower()
            operation = self.morph_operation_combo.currentText().lower().replace(" ", "_")
            strength = self.morph_strength_slider.value()
            preserve_colors = self.morph_preserve_colors_checkbox.isChecked()
            
            # Apply smoothing
            start_time = time.time()
            result = self._apply_custom_morphological_smoothing(
                self.original_image, kernel_size, kernel_shape, operation, strength, preserve_colors
            )
            end_time = time.time()
            
            # Show results
            param_info = f"{kernel_shape} {operation} size={kernel_size} strength={strength}"
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
        
        # Convert to grayscale for morphological operations
        gray = cv.cvtColor(rgba[:, :, :3], cv.COLOR_RGB2GRAY)
        
        # Create kernel
        if kernel_shape == "rectangle":
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == "ellipse":
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        elif kernel_shape == "cross":
            kernel = cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
        else:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
        
        # Apply operation
        if operation == "opening":
            result_gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations=strength)
        elif operation == "closing":
            result_gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel, iterations=strength)
        elif operation == "gradient":
            result_gray = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel, iterations=strength)
        elif operation == "top_hat":
            result_gray = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel, iterations=strength)
        elif operation == "black_hat":
            result_gray = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel, iterations=strength)
        else:
            result_gray = gray.copy()
        
        # Convert back to RGB
        result_rgb = cv.cvtColor(result_gray, cv.COLOR_GRAY2RGB)
        
        # Create result with original alpha
        result = rgba.copy()
        if preserve_colors:
            # Keep original colors where alpha > 0
            mask = rgba[:, :, 3] > 0
            result[mask, :3] = result_rgb[mask]
        else:
            result[:, :, :3] = result_rgb
            
        return result
        
    def apply_boundary_smoothing(self):
        """Apply tendril trimming."""
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        try:
            # Get parameters
            gaussian_kernel = self.gaussian_kernel_spinbox.value()
            unlimited_iterations = self.unlimited_iterations_checkbox.isChecked()
            tendril_threshold = self.tendril_threshold_spinbox.value()
            
            # Debug: Show what parameters are being used
            self.progress_label.setText(f"Parameters: unlimited={unlimited_iterations}, threshold={tendril_threshold}")
            
            # Update progress
            self.progress_label.setText("Applying tendril trimming...")
            self.statusBar().showMessage("Processing tendril trimming...")
            
            # Apply tendril trimming with progress tracking
            start_time = time.time()
            step_mode = self.step_mode_checkbox.isChecked()
            
            # Create tendril algorithm instance
            tendril_algorithm = TendrilAlgorithm()
            tendril_algorithm.progress_updated.connect(self.progress_label.setText)
            tendril_algorithm.iteration_completed.connect(self.on_iteration_completed)
            
            result = tendril_algorithm.apply_tendril_trimming_with_progress(
                self.original_image, 
                gaussian_kernel, unlimited_iterations, tendril_threshold, step_mode
            )
            end_time = time.time()
            
            # Show results with parameter info
            param_parts = ["Tendril-Trimming"]
            param_parts.append(f"TEND:{tendril_threshold}")
            if not unlimited_iterations:
                param_parts.append(f"MAX:{gaussian_kernel}")
            
            param_info = " ".join(param_parts)
            self.show_boundary_results(result, end_time - start_time, param_info)
            
            # Update progress
            self.progress_label.setText(f"✓ Tendril trimming completed in {end_time - start_time:.3f}s")
            self.statusBar().showMessage("Tendril trimming completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Tendril Trimming Error", f"Failed to apply tendril trimming:\n{str(e)}")
            self.progress_label.setText("Error applying tendril trimming")
            
    def on_iteration_completed(self, iteration, tendrils_found, pixels_recolored):
        """Handle iteration completion."""
        self.statusBar().showMessage(f"Iteration {iteration}: {tendrils_found} tendrils found, {pixels_recolored} pixels recolored")
        
    def show_morphological_results(self, result_image, time_taken, param_info=""):
        """Show morphological smoothing results."""
        # Convert to QPixmap
        qimage = numpy_rgba_to_qimage(result_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set in view
        self.morph_result_view.set_pixmap(pixmap)
        
    def show_boundary_results(self, result_image, time_taken, param_info=""):
        """Show boundary smoothing results while preserving zoom/pan state."""
        # If this is the first time, create the views
        if self.boundary_result_view is None:
            self.boundary_result_view = SynchronizedImageView()
            
        # Convert to QPixmap
        qimage = numpy_rgba_to_qimage(result_image)
        pixmap = QPixmap.fromImage(qimage)
        
        # Set in view
        self.boundary_result_view.set_pixmap(pixmap)
