from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal, QSize
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, 
    QGroupBox, QFormLayout, QCheckBox, QSpinBox, QTextEdit, QScrollArea,
    QDialog, QDialogButtonBox, QGridLayout, QMessageBox, QComboBox
)

from model import AppState
from processing.color_simplify import create_palette_from_colors
from ui.base_step import BaseStep
from ui.region_cleanup.region_merge_dialog import RegionMergeDialog, ColorSwatch
from ui.region_cleanup.flood_fill_color_dialog import FloodFillColorDialog, create_flood_fill_icon
from ui.region_cleanup.flood_fill_view import FloodFillView

class RegionCleanupPanel(BaseStep):
    """Panel for region cleanup and merging operations."""
    
    # TODO: Limit Region Cleanup step to images with 32 or fewer colors for performance reasons.
    # Images with more colors can cause significant performance issues during region analysis.
    
    statusBarMessage = Signal(str)
    
    def __init__(self, parent: Optional[QWidget] = None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None) -> None:
        super().__init__(parent, app_state, image_view)
        self._min_region_size = 100
        self._region_stats: Dict = {}
        self._selected_flood_fill_color: Optional[QColor] = None
        
        # Create flood fill view
        self._flood_fill_view = FloodFillView(self._image_view, self._app_state, self._image_view)
        
        # Set maximum height to prevent the panel from being too tall
        self.setMaximumHeight(600)
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        # Create a scroll area to contain the main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(600)  # Limit the height of the scroll area
        
        # Create the main content widget
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("Region Cleanup")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Remove small regions that are too small for laser engraving by merging them "
            "into larger neighboring regions. Use flood-fill to manually adjust regions."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Flood Fill Tool (moved to top)
        flood_fill_group = QGroupBox("Flood Fill Tool")
        flood_fill_layout = QHBoxLayout(flood_fill_group)
        
        # Enable flood fill checkbox
        self.flood_fill_enabled = QCheckBox("Enable Flood Fill")
        self.flood_fill_enabled.setToolTip("Enable flood fill mode to paint regions with the selected color")
        flood_fill_layout.addWidget(self.flood_fill_enabled)
        
        # Color selection button
        color_layout = QHBoxLayout()
        self.flood_fill_button = QPushButton("Choose Color")
        # Set initial icon (will be updated when color is selected)
        self.flood_fill_button.setIcon(create_flood_fill_icon())
        self.flood_fill_button.setIconSize(QSize(24, 24))
        color_layout.addWidget(self.flood_fill_button)
        color_layout.addStretch()
        flood_fill_layout.addLayout(color_layout)
        
        layout.addWidget(flood_fill_group)
        
        # Region size threshold (consolidated section)
        threshold_group = QGroupBox("Region Size Threshold")
        threshold_layout = QFormLayout(threshold_group)
        
        # Minimum region size input field
        self.size_input = QSpinBox()
        self.size_input.setMinimum(10)
        self.size_input.setMaximum(1000)
        self.size_input.setValue(100)
        self.size_input.setSuffix(" pixels")
        threshold_layout.addRow("Minimum Region Size", self.size_input)
        
        # Merge regions button (moved here)
        self.cleanup_button = QPushButton("Merge Regions")
        threshold_layout.addRow("", self.cleanup_button)
        
        layout.addWidget(threshold_group)
        
        # Region statistics
        stats_group = QGroupBox("Region Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(120)
        self.stats_text.setReadOnly(True)
        self.stats_text.setText("No region analysis available")
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        # Smoothing Section (renamed from Tendril Cleanup)
        smoothing_group = QGroupBox("Smoothing")
        smoothing_layout = QFormLayout()
        
        # Smoothing strength slider with label
        self.tendril_threshold_slider = QSlider(Qt.Horizontal)
        self.tendril_threshold_slider.setRange(1, 10)
        self.tendril_threshold_slider.setValue(2)  # Default to 2
        self.tendril_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.tendril_threshold_slider.setTickInterval(1)
        self.tendril_threshold_slider.setToolTip("Strength: 1-10 pixels")
        
        self.tendril_threshold_label = QLabel("2")
        self.tendril_threshold_label.setMinimumWidth(30)
        self.tendril_threshold_label.setAlignment(Qt.AlignCenter)
        
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.tendril_threshold_slider)
        strength_layout.addWidget(self.tendril_threshold_label)
        
        smoothing_layout.addRow("Strength (pixels)", strength_layout)
        
        # Max iterations slider with label
        self.tendril_iterations_slider = QSlider(Qt.Horizontal)
        self.tendril_iterations_slider.setRange(5, 100)
        self.tendril_iterations_slider.setValue(20)
        self.tendril_iterations_slider.setTickPosition(QSlider.TicksBelow)
        self.tendril_iterations_slider.setTickInterval(10)
        self.tendril_iterations_slider.setToolTip("Max Iterations: 5-100")
        
        self.tendril_iterations_label = QLabel("20")
        self.tendril_iterations_label.setMinimumWidth(30)
        self.tendril_iterations_label.setAlignment(Qt.AlignCenter)
        
        iterations_layout = QHBoxLayout()
        iterations_layout.addWidget(self.tendril_iterations_slider)
        iterations_layout.addWidget(self.tendril_iterations_label)
        
        smoothing_layout.addRow("Max Iterations", iterations_layout)
        
        # Smoothing button
        self.tendril_cleanup_button = QPushButton("Apply Smoothing")
        smoothing_layout.addRow("", self.tendril_cleanup_button)
        
        smoothing_group.setLayout(smoothing_layout)
        layout.addWidget(smoothing_group)
        
        # Set the scroll area's widget and use it as the main widget
        scroll_area.setWidget(main_widget)
        self.set_main_widget(scroll_area)
    
    def get_min_region_size(self) -> int:
        """Get the minimum region size threshold."""
        return self.size_input.value()
    
    def update_region_statistics(self, stats: Dict) -> None:
        """Update the region statistics display."""
        self._region_stats = stats
        
        if not stats:
            self.stats_text.setText("No region analysis available")
            return
        
        text = f"Total Regions: {stats.get('total_regions', 0)}\n"
        text += f"Regions below threshold: {stats.get('small_regions', 0)}\n"
        text += f"Largest region: {stats.get('largest_region_size', 0)} pixels\n"
        text += f"Smallest region: {stats.get('smallest_region_size', 0)} pixels\n"
        
        if 'size_distribution' in stats:
            text += "\nSize Distribution:\n"
            for size_range, count in stats['size_distribution'].items():
                text += f"  {size_range}: {count} regions\n"
        
        self.stats_text.setText(text)
    
    def show_merge_dialog(self, small_region_color: QColor, neighbor_colors: List[QColor], image_data: np.ndarray = None, bbox: Tuple[int, int, int, int] = None) -> Optional[QColor]:
        """Show dialog for choosing merge color and return selected color."""
        dialog = RegionMergeDialog(small_region_color, neighbor_colors, self, image_data, bbox)
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_selected_color()
        return None
    
    def get_selected_flood_fill_color(self) -> Optional[QColor]:
        """Get the currently selected flood fill color."""
        return getattr(self, '_selected_flood_fill_color', None)
    
    def _on_flood_fill_button_clicked(self) -> None:
        """Handle flood fill color selection button click."""
        # Get unique colors from the current working image
        working_image = self._app_state.working_image
        if working_image is None:
            QMessageBox.warning(self, "Flood Fill", "No image loaded. Please load an image first.")
            return
        
        # Extract unique colors from the image
        unique_colors = self._extract_unique_colors(working_image)
        
        if not unique_colors:
            QMessageBox.warning(self, "Flood Fill", "No colors found in the image.")
            return
        
        # Show color selection dialog
        dialog = FloodFillColorDialog(unique_colors, self)
        if dialog.exec() == QDialog.Accepted:
            selected_color = dialog.get_selected_color()
            if selected_color is not None:
                self._selected_flood_fill_color = selected_color
                # Update button icon with selected color
                self.flood_fill_button.setIcon(create_flood_fill_icon(selected_color))
                # Update flood fill view color if it's active
                if self._flood_fill_view._active:
                    self._flood_fill_view.set_fill_color(selected_color)
                # Enable flood fill checkbox after color selection
                self.flood_fill_enabled.setChecked(True)

    def _on_flood_fill_enabled_toggled(self, enabled: bool) -> None:
        """Handle flood fill enable/disable toggle."""
        if enabled:
            if self._selected_flood_fill_color is None:
                QMessageBox.warning(self, "Flood Fill", "Please select a color first by clicking the 'Choose Color' button.")
                self.flood_fill_enabled.setChecked(False)
                return
            # Activate flood fill view
            self._flood_fill_view.set_fill_color(self._selected_flood_fill_color)
            self._flood_fill_view.set_active(True)
            self.statusBarMessage.emit("Flood fill mode enabled - click on regions to fill them")
        else:
            # Deactivate flood fill view
            self._flood_fill_view.set_active(False)
            self.statusBarMessage.emit("Flood fill mode disabled")

    def _on_flood_fill_requested(self, position: QPoint, fill_color: QColor) -> None:
        """Handle flood fill request from the flood fill view."""
        working_image = self._app_state.working_image
        if working_image is None:
            QMessageBox.warning(self, "Flood Fill", "No image available for flood fill.")
            return
        
        # Perform flood fill
        try:
            from processing.region_cleanup import flood_fill_region
            
            # Convert QColor to RGB tuple
            fill_color_rgb = (fill_color.red(), fill_color.green(), fill_color.blue())
            
            # Perform flood fill
            filled_rgba = flood_fill_region(working_image, (position.x(), position.y()), fill_color_rgb)
            
            if filled_rgba is not None:
                # Update the working image with the filled result
                self.set_working_image(filled_rgba)
                
                # Show success message
                self.statusBarMessage.emit(f"Flood fill applied with color {fill_color.name()}")
            else:
                self.statusBarMessage.emit("Flood fill failed - no region found at the specified position")
                
        except Exception as e:
            QMessageBox.warning(self, "Flood Fill", f"Flood fill failed: {e}")
            self.statusBarMessage.emit(f"Flood fill error: {e}")

    def _extract_unique_colors(self, image: np.ndarray) -> List[QColor]:
        """Extract unique colors from the image."""
        if image is None or len(image.shape) < 3:
            return []
        
        # Reshape image to get all pixels
        pixels = image.reshape(-1, image.shape[-1])
        
        # Get unique colors (only RGB channels, ignore alpha)
        if pixels.shape[1] >= 3:
            rgb_pixels = pixels[:, :3]
            
            # Filter out transparent pixels if alpha channel exists
            if pixels.shape[1] >= 4:
                alpha = pixels[:, 3]
                # Only include pixels with alpha == 255 (opaque)
                visible_mask = alpha == 255
                rgb_pixels = rgb_pixels[visible_mask]
            
            if len(rgb_pixels) == 0:
                return []
            
            unique_pixels = np.unique(rgb_pixels, axis=0)
            
            # Convert to QColor list and filter out common background colors
            colors = []
            for pixel in unique_pixels:
                r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
                
                color = QColor(r, g, b)
                colors.append(color)
            
            return colors
        
        return []

    def _on_region_cleanup_button_clicked(self) -> None:
        """Handle region cleanup button click."""
        # Check if we have a processed image available
        working_image = self._app_state.working_image
        if working_image is None:
            QMessageBox.warning(self, "Region Cleanup", "No image available for processing.")
            return
        
        # Get parameters from the panel
        min_size = self.get_min_region_size()
        auto_merge_threshold = 0; # Default 0, self.get_auto_merge_threshold()
        connectivity = 8; # Default 8, self.get_connectivity()
        
        # Show progress dialog
        from ui.progress_dialog import ProgressDialog
        progress_dialog = ProgressDialog("Region Cleanup", self)
        progress_dialog.show()
        
        try:
            # Perform region cleanup
            from processing.region_cleanup import merge_small_regions
            
            def progress_callback(current: int, total: int, message: str) -> None:
                progress_dialog.update_progress(current, total, message)
            
            cleaned_output = merge_small_regions(
                working_image,
                min_size,
                merge_callback=None,  # Auto-merge for now
                auto_merge_threshold=auto_merge_threshold,
                progress_callback=progress_callback,
                connectivity=connectivity
            )
            
            if cleaned_output is not None:
                # Update the working image with the cleaned result
                self.set_working_image(cleaned_output)
                
                # Update statistics
                from processing.region_cleanup import analyze_regions
                stats = analyze_regions(cleaned_output, min_size)
                self.update_region_statistics(stats)
                
                self.statusBarMessage.emit("Region cleanup completed successfully.")
            else:
                QMessageBox.information(self, "Region Cleanup", "Region cleanup was cancelled.")
        
        except Exception as e:
            QMessageBox.critical(self, "Region Cleanup Error", f"An error occurred during region cleanup:\n{str(e)}")
        
        finally:
            progress_dialog.close()

    def _on_tendril_cleanup_button_clicked(self) -> None:
        """Handle tendril cleanup button click."""
        # Check if a worker is already running
        if hasattr(self, '_tendril_worker') and self._tendril_worker and self._tendril_worker.isRunning():
            QMessageBox.warning(self, "Smoothing", "A smoothing operation is already in progress.")
            return
        
        # Check if we have a processed image available
        working_image = self._app_state.working_image
        if working_image is None:
            QMessageBox.warning(self, "Smoothing", "No image available for processing.")
            return
        
        # Get parameters from the panel
        threshold = self.get_tendril_threshold()
        max_iterations = self.get_tendril_max_iterations()
        
        # Show progress dialog
        from ui.progress_dialog import ProgressDialog
        progress_dialog = ProgressDialog("Smoothing", self)
        progress_dialog.show()
        
        try:
            # Create and start worker thread
            from ui.tendril_worker import TendrilWorker
            self._tendril_worker = TendrilWorker(working_image, threshold, max_iterations)
            
            # Store progress dialog as instance variable to avoid scope issues
            self._progress_dialog = progress_dialog
            
            # Connect signals
            self._tendril_worker.progress_updated.connect(progress_dialog.update_progress)
            self._tendril_worker.cleanup_completed.connect(self._on_tendril_cleanup_completed)
            self._tendril_worker.cleanup_failed.connect(self._on_tendril_cleanup_failed)
            
            # Only start if not already running
            if not self._tendril_worker.isRunning():
                self._tendril_worker.start()
            
        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(self, "Smoothing Error", f"Failed to start smoothing process:\n{str(e)}")
    
    def _on_tendril_cleanup_completed(self, cleaned_output: np.ndarray, iterations_used: int, status_message: str) -> None:
        """Handle successful tendril cleanup completion."""
        try:
            # Close progress dialog safely
            if hasattr(self, '_progress_dialog') and self._progress_dialog:
                try:
                    self._progress_dialog.close()
                except Exception:
                    pass  # Ignore any errors closing the dialog
            
            # Disconnect signals to prevent memory leaks
            if hasattr(self, '_tendril_worker') and self._tendril_worker:
                self._tendril_worker.progress_updated.disconnect()
                self._tendril_worker.cleanup_completed.disconnect()
                self._tendril_worker.cleanup_failed.disconnect()
                
                # Wait for thread to finish before cleanup
                if self._tendril_worker.isRunning():
                    self._tendril_worker.wait(2000)  # Wait up to 2 seconds
                    if self._tendril_worker.isRunning():
                        self._tendril_worker.terminate()
                        self._tendril_worker.wait(1000)  # Wait for termination
                
                self._tendril_worker = None
            
            # Update the working image with the cleaned result
            self.set_working_image(cleaned_output)
            
            # Show success message
            self.statusBarMessage.emit(f"Smoothing completed after {iterations_used} iterations")
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"An error occurred during completion:\n{str(e)}")
    
    def _on_tendril_cleanup_failed(self, error_message: str) -> None:
        """Handle tendril cleanup failure."""
        try:
            # Close progress dialog safely
            if hasattr(self, '_progress_dialog') and self._progress_dialog:
                try:
                    self._progress_dialog.close()
                except Exception:
                    pass  # Ignore any errors closing the dialog
            
            # Disconnect signals to prevent memory leaks
            if hasattr(self, '_tendril_worker') and self._tendril_worker:
                self._tendril_worker.progress_updated.disconnect()
                self._tendril_worker.cleanup_completed.disconnect()
                self._tendril_worker.cleanup_failed.disconnect()
                # Don't set to None immediately - let Qt handle cleanup
            
            QMessageBox.critical(self, "Smoothing Error", f"An error occurred during smoothing:\n{error_message}")
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"An error occurred during failure handling:\n{str(e)}")
    
    def _on_save_requested(self) -> None:
        """Handle save button click."""
        self.saveRequested.emit()
    
    def _on_tendril_threshold_changed(self, value: int) -> None:
        """Handle smoothing strength slider change."""
        self.tendril_threshold_label.setText(f"{value}")
    
    def _on_tendril_iterations_changed(self, value: int) -> None:
        """Handle tendril iterations slider change."""
        self.tendril_iterations_label.setText(f"{value}")
    
    def get_tendril_threshold(self) -> int:
        """Get the tendril thickness threshold."""
        return self.tendril_threshold_slider.value()
    
    def get_tendril_max_iterations(self) -> int:
        """Get the maximum number of tendril cleanup iterations."""
        return self.tendril_iterations_slider.value()

    def validate_entry(self) -> bool:
        """Validate the entry of the step."""
        # Check if there is a working image
        working_image = self._app_state.working_image
        if working_image is None:
            QMessageBox.warning(
                self, 
                "Region Cleanup Validation", 
                "No working image available.\n\nPlease load an image and process it through the previous steps before using Region Cleanup."
            )
            return False
        
        # Check if the number of colors in the palette is no more than 32
        try:
            unique_colors = self._extract_unique_colors(working_image)
            num_colors = len(unique_colors)
            
            if num_colors > 32:
                QMessageBox.warning(
                    self,
                    "Region Cleanup Validation",
                    f"Image has too many colors ({num_colors}).\n\nRegion Cleanup works best with images that have 32 or fewer colors.\n\nPlease use the Color Processing step to reduce the number of colors first."
                )
                return False
            
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Region Cleanup Validation",
                f"Error analyzing image: {str(e)}\n\nPlease ensure the image is properly loaded and try again."
            )
            return False

    def _on_open(self) -> None:
        """Handle open event."""
        super()._on_open()
        self.tendril_threshold_slider.valueChanged.connect(self._on_tendril_threshold_changed)
        self.tendril_iterations_slider.valueChanged.connect(self._on_tendril_iterations_changed)
        self.flood_fill_button.clicked.connect(self._on_flood_fill_button_clicked)
        self.flood_fill_enabled.toggled.connect(self._on_flood_fill_enabled_toggled)
        self._flood_fill_view.flood_fill_requested.connect(self._on_flood_fill_requested)
        self.cleanup_button.clicked.connect(self._on_region_cleanup_button_clicked)
        self.tendril_cleanup_button.clicked.connect(self._on_tendril_cleanup_button_clicked)

    def _on_close(self) -> None:
        """Handle close event."""
        super()._on_close()
        self.tendril_threshold_slider.valueChanged.disconnect(self._on_tendril_threshold_changed)
        self.tendril_iterations_slider.valueChanged.disconnect(self._on_tendril_iterations_changed)
        self.flood_fill_button.clicked.disconnect(self._on_flood_fill_button_clicked)
        self.flood_fill_enabled.toggled.disconnect(self._on_flood_fill_enabled_toggled)
        self._flood_fill_view.flood_fill_requested.disconnect(self._on_flood_fill_requested)
        self.cleanup_button.clicked.disconnect(self._on_region_cleanup_button_clicked)
        self.tendril_cleanup_button.clicked.disconnect(self._on_tendril_cleanup_button_clicked)
        
        # Clean up any running tendril worker
        if hasattr(self, '_tendril_worker') and self._tendril_worker:
            self._tendril_worker.progress_updated.disconnect()
            self._tendril_worker.cleanup_completed.disconnect()
            self._tendril_worker.cleanup_failed.disconnect()
            self._tendril_worker = None
        
        # Disable flood fill mode when closing
        self.flood_fill_enabled.setChecked(False)
        self._flood_fill_view.set_active(False)
