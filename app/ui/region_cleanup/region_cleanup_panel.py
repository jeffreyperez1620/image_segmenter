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
        
        # Minimum region size slider
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(10)
        self.size_slider.setMaximum(1000)
        self.size_slider.setValue(100)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(100)
        self.size_slider.valueChanged.connect(self._on_size_threshold_changed)
        threshold_layout.addRow("Minimum Region Size (pixels)", self.size_slider)
        
        self.size_label = QLabel("100 pixels")
        threshold_layout.addRow("Current threshold:", self.size_label)
        
        # Auto-merge threshold (moved here)
        self.auto_merge_threshold_slider = QSlider(Qt.Horizontal)
        self.auto_merge_threshold_slider.setMinimum(0)
        self.auto_merge_threshold_slider.setMaximum(100)
        self.auto_merge_threshold_slider.setValue(0)  # Default to 0%
        self.auto_merge_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.auto_merge_threshold_slider.setTickInterval(10)
        self.auto_merge_threshold_slider.setToolTip("Confidence threshold for automatic merging (0-100%). 0% is completely automatic, 100% is completely manual.")
        threshold_layout.addRow("Auto-merge Threshold (%)", self.auto_merge_threshold_slider)
        
        self.auto_merge_label = QLabel("0%")
        threshold_layout.addRow("Current threshold:", self.auto_merge_label)
        self.auto_merge_threshold_slider.valueChanged.connect(self._on_auto_merge_threshold_changed)
        
        # Note about auto-merge threshold
        auto_merge_note = QLabel("Note: 0% is completely automatic, 100% is completely manual")
        auto_merge_note.setStyleSheet("color: gray; font-size: 11px;")
        threshold_layout.addRow("", auto_merge_note)
        
        # Adjacency method (moved here)
        self.connectivity_combo = QComboBox()
        self.connectivity_combo.addItems(["8-way (diagonal)", "4-way (horizontal/vertical only)"])
        self.connectivity_combo.setCurrentIndex(0)  # Default to 8-way
        self.connectivity_combo.setToolTip("8-way: regions connected by diagonal pixels are merged. 4-way: only horizontal/vertical connections.")
        threshold_layout.addRow("Adjacency Method:", self.connectivity_combo)
        
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
        
        # Visualization options
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)
        
        layout.addWidget(viz_group)
        
        # Smoothing Section (renamed from Tendril Cleanup)
        smoothing_group = QGroupBox("Smoothing")
        smoothing_layout = QFormLayout()
        
        # Smoothing strength slider (renamed from Tendril Thickness Threshold)
        self.tendril_threshold_slider = QSlider(Qt.Horizontal)
        self.tendril_threshold_slider.setRange(1, 10)
        self.tendril_threshold_slider.setValue(2)  # Default to 2
        self.tendril_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.tendril_threshold_slider.setTickInterval(1)
        self.tendril_threshold_slider.valueChanged.connect(self._on_tendril_threshold_changed)
        smoothing_layout.addRow("Strength", self.tendril_threshold_slider)
        
        self.tendril_threshold_label = QLabel("2 pixels")
        smoothing_layout.addRow("Current setting:", self.tendril_threshold_label)
        
        # Max iterations slider
        self.tendril_iterations_slider = QSlider(Qt.Horizontal)
        self.tendril_iterations_slider.setRange(5, 50)
        self.tendril_iterations_slider.setValue(30)
        self.tendril_iterations_slider.setTickPosition(QSlider.TicksBelow)
        self.tendril_iterations_slider.setTickInterval(5)
        self.tendril_iterations_slider.valueChanged.connect(self._on_tendril_iterations_changed)
        smoothing_layout.addRow("Max Iterations", self.tendril_iterations_slider)
        
        self.tendril_iterations_label = QLabel("30 iterations")
        smoothing_layout.addRow("Current setting:", self.tendril_iterations_label)
        
        # Smoothing button
        self.tendril_cleanup_button = QPushButton("Apply Smoothing")
        self.tendril_cleanup_button.clicked.connect(self._on_tendril_cleanup_button_clicked)
        smoothing_layout.addRow("", self.tendril_cleanup_button)
        
        # Instructions (updated text)
        smoothing_instructions = QLabel(
            "Remove thin tendrils (pixels with thickness ≤ strength) that are too thin for laser engraving. "
            "The algorithm will iterate until no more tendrils are found or max iterations is reached."
        )
        smoothing_instructions.setWordWrap(True)
        smoothing_instructions.setStyleSheet("color: gray; font-size: 11px;")
        smoothing_layout.addRow("", smoothing_instructions)
        
        smoothing_group.setLayout(smoothing_layout)
        layout.addWidget(smoothing_group)
        
        # Set the scroll area's widget and use it as the main widget
        scroll_area.setWidget(main_widget)
        self.set_main_widget(scroll_area)
    
    def _on_size_threshold_changed(self, value: int) -> None:
        """Handle size threshold slider change."""
        self._min_region_size = value
        self.size_label.setText(f"{value} pixels")
    
    def _on_auto_merge_threshold_changed(self, value: int) -> None:
        """Handle auto-merge threshold slider change."""
        self.auto_merge_label.setText(f"{value}%")
    
    def get_min_region_size(self) -> int:
        """Get the minimum region size threshold."""
        return self._min_region_size
    
    def get_auto_merge_threshold(self) -> float:
        """Get the auto-merge threshold as a float between 0 and 1."""
        return self.auto_merge_threshold_slider.value() / 100.0
    
    def get_connectivity(self) -> int:
        """Get the connectivity setting (4 or 8)."""
        return 4 if self.connectivity_combo.currentIndex() == 1 else 8
    
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
        auto_merge_threshold = self.get_auto_merge_threshold()
        connectivity = self.get_connectivity()
        
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
    
    def _on_save_requested(self) -> None:
        """Handle save button click."""
        self.saveRequested.emit()
    
    def _on_tendril_threshold_changed(self, value: int) -> None:
        """Handle smoothing strength slider change."""
        self.tendril_threshold_label.setText(f"{value} pixels")
    
    def _on_tendril_iterations_changed(self, value: int) -> None:
        """Handle tendril iterations slider change."""
        self.tendril_iterations_label.setText(f"{value} iterations")
    
    def get_tendril_threshold(self) -> int:
        """Get the tendril thickness threshold."""
        return self.tendril_threshold_slider.value()
    
    def get_tendril_max_iterations(self) -> int:
        """Get the maximum number of tendril cleanup iterations."""
        return self.tendril_iterations_slider.value()

    def _on_open(self) -> None:
        """Handle open event."""
        super()._on_open()
        self.flood_fill_button.clicked.connect(self._on_flood_fill_button_clicked)
        self.flood_fill_enabled.toggled.connect(self._on_flood_fill_enabled_toggled)
        self._flood_fill_view.flood_fill_requested.connect(self._on_flood_fill_requested)
        self.cleanup_button.clicked.connect(self._on_region_cleanup_button_clicked)

    def _on_close(self) -> None:
        """Handle close event."""
        super()._on_close()
        self.flood_fill_button.clicked.disconnect(self._on_flood_fill_button_clicked)
        self.flood_fill_enabled.toggled.disconnect(self._on_flood_fill_enabled_toggled)
        self._flood_fill_view.flood_fill_requested.disconnect(self._on_flood_fill_requested)
        self.cleanup_button.clicked.disconnect(self._on_region_cleanup_button_clicked)
        # Disable flood fill mode when closing
        self.flood_fill_enabled.setChecked(False)
        self._flood_fill_view.set_active(False)
