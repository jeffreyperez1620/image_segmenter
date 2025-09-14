from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, 
    QGroupBox, QFormLayout, QCheckBox, QSpinBox, QTextEdit, QScrollArea,
    QDialog, QDialogButtonBox, QGridLayout, QMessageBox, QComboBox
)

from processing.color_simplify import create_palette_from_colors


class ColorSwatch(QLabel):
    """A simple color swatch widget for displaying colors."""
    
    colorSelected = Signal(QColor)
    
    def __init__(self, color: QColor, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.color = color
        self.setFixedSize(30, 30)
        self.setFrameStyle(1)  # Box frame
        self.setLineWidth(2)
        self.update_display()
    
    def update_display(self) -> None:
        """Update the visual display of the color swatch."""
        from PySide6.QtGui import QPixmap, QPainter, QPen
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.fillRect(pixmap.rect(), self.color)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(pixmap.rect().adjusted(1, 1, -1, -1))
        painter.end()
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """Handle mouse clicks to select color."""
        if event.button() == Qt.LeftButton:
            self.colorSelected.emit(self.color)


class RegionMergeDialog(QDialog):
    """Dialog for choosing which color to merge a small region into."""
    
    def __init__(self, small_region_color: QColor, neighbor_colors: List[QColor], parent: Optional[QWidget] = None, image_data: np.ndarray = None, bbox: Tuple[int, int, int, int] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Merge Small Region")
        self.setModal(True)
        self.resize(600, 500)
        
        self.selected_color: Optional[QColor] = None
        self.image_data = image_data
        self.bbox = bbox
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            f"The region with color RGB({small_region_color.red()}, {small_region_color.green()}, {small_region_color.blue()}) "
            f"is too small and needs to be merged. Choose which neighboring color to merge it into:"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Show image preview if available
        if self.image_data is not None and self.bbox is not None:
            preview_label = QLabel("Region Preview:")
            layout.addWidget(preview_label)
            
            # Create cropped image preview
            x, y, w, h = self.bbox
            cropped_image = self.image_data[y:y+h, x:x+w]
            
            # Convert to QImage and display - ensure array is contiguous
            from utils.qt_image import numpy_rgba_to_qimage
            import numpy as np
            cropped_image_contiguous = np.ascontiguousarray(cropped_image)
            qimg = numpy_rgba_to_qimage(cropped_image_contiguous)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale the image to a good viewing size
            # If too large, scale down; if too small, scale up
            if pixmap.width() > 400 or pixmap.height() > 400:
                pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            elif pixmap.width() < 100 or pixmap.height() < 100:
                # Scale up small regions so they're visible
                scale_factor = max(100 / pixmap.width(), 100 / pixmap.height())
                new_width = int(pixmap.width() * scale_factor)
                new_height = int(pixmap.height() * scale_factor)
                pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            preview_image_label = QLabel()
            preview_image_label.setPixmap(pixmap)
            preview_image_label.setAlignment(Qt.AlignCenter)
            preview_image_label.setStyleSheet("border: 2px solid black;")
            layout.addWidget(preview_image_label)
        
        # Show the small region color
        small_region_layout = QHBoxLayout()
        small_region_layout.addWidget(QLabel("Small region:"))
        small_swatch = ColorSwatch(small_region_color)
        small_swatch.setEnabled(False)
        small_region_layout.addWidget(small_swatch)
        small_region_layout.addStretch()
        layout.addLayout(small_region_layout)
        
        # Show neighbor colors
        neighbors_label = QLabel("Choose neighboring color to merge into:")
        layout.addWidget(neighbors_label)
        
        # Create a grid of neighbor color swatches
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        cols = 6
        for i, color in enumerate(neighbor_colors):
            swatch = ColorSwatch(color)
            swatch.colorSelected.connect(self._on_color_selected)
            row = i // cols
            col = i % cols
            scroll_layout.addWidget(swatch, row, col)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMaximumHeight(150)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        layout.addWidget(button_box)
    
    def _on_color_selected(self, color: QColor) -> None:
        """Handle color selection."""
        self.selected_color = color
        self.ok_button.setEnabled(True)
    
    def get_selected_color(self) -> Optional[QColor]:
        """Get the selected color for merging."""
        return self.selected_color


class RegionCleanupPanel(QWidget):
    """Panel for region cleanup and merging operations."""
    
    cleanupRequested = Signal()
    previewToggled = Signal(bool)
    applyRequested = Signal()
    floodFillRequested = Signal()
    regionBoundariesToggled = Signal(bool)
    smoothingRequested = Signal(str, float, bool)  # method, strength, preserve_colors
    saveRequested = Signal()  # Signal to save working image
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._min_region_size = 100
        self._show_region_overlay = False
        self._region_stats: Dict = {}
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
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
        
        # Region size threshold
        threshold_group = QGroupBox("Region Size Threshold")
        threshold_layout = QFormLayout(threshold_group)
        
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
        
        layout.addWidget(threshold_group)
        
        # Auto-merge settings
        auto_merge_group = QGroupBox("Auto-Merge Settings")
        auto_merge_layout = QFormLayout(auto_merge_group)
        
        self.auto_merge_threshold_slider = QSlider(Qt.Horizontal)
        self.auto_merge_threshold_slider.setMinimum(0)
        self.auto_merge_threshold_slider.setMaximum(100)
        self.auto_merge_threshold_slider.setValue(70)
        self.auto_merge_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.auto_merge_threshold_slider.setTickInterval(10)
        self.auto_merge_threshold_slider.setToolTip("Confidence threshold for automatic merging (0-100%). Higher values require more user decisions.")
        auto_merge_layout.addRow("Auto-merge Threshold (%)", self.auto_merge_threshold_slider)
        
        self.auto_merge_label = QLabel("70%")
        auto_merge_layout.addRow("Current threshold:", self.auto_merge_label)
        self.auto_merge_threshold_slider.valueChanged.connect(self._on_auto_merge_threshold_changed)
        
        layout.addWidget(auto_merge_group)
        
        # Connectivity settings
        connectivity_group = QGroupBox("Connectivity Settings")
        connectivity_layout = QFormLayout(connectivity_group)
        
        self.connectivity_combo = QComboBox()
        self.connectivity_combo.addItems(["8-way (diagonal)", "4-way (horizontal/vertical only)"])
        self.connectivity_combo.setCurrentIndex(0)  # Default to 8-way
        self.connectivity_combo.setToolTip("8-way: regions connected by diagonal pixels are merged. 4-way: only horizontal/vertical connections.")
        connectivity_layout.addRow("Adjacency Method:", self.connectivity_combo)
        
        layout.addWidget(connectivity_group)
        
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
        
        self.show_overlay_checkbox = QCheckBox("Show Region Boundaries")
        self.show_overlay_checkbox.toggled.connect(self._on_overlay_toggled)
        viz_layout.addRow("", self.show_overlay_checkbox)
        
        layout.addWidget(viz_group)
        
        # Flood Fill Tool
        flood_fill_group = QGroupBox("Flood Fill Tool")
        flood_fill_layout = QVBoxLayout(flood_fill_group)
        
        # Color palette for flood fill
        palette_label = QLabel("Select flood fill color:")
        flood_fill_layout.addWidget(palette_label)
        
        # Color palette scroll area
        self.palette_scroll = QScrollArea()
        self.palette_scroll.setMaximumHeight(120)
        self.palette_scroll.setWidgetResizable(True)
        self.palette_widget = QWidget()
        self.palette_layout = QGridLayout(self.palette_widget)
        self.palette_layout.setSpacing(2)
        self.palette_scroll.setWidget(self.palette_widget)
        flood_fill_layout.addWidget(self.palette_scroll)
        
        # Selected color display
        selected_color_layout = QHBoxLayout()
        selected_color_layout.addWidget(QLabel("Selected:"))
        self.selected_color_swatch = ColorSwatch(QColor(128, 128, 128))  # Default gray
        self.selected_color_swatch.setFixedSize(40, 40)
        # Don't disable the swatch - it should show full color
        selected_color_layout.addWidget(self.selected_color_swatch)
        selected_color_layout.addStretch()
        flood_fill_layout.addLayout(selected_color_layout)
        
        # Flood fill button
        self.flood_fill_button = QPushButton("Activate Flood Fill")
        self.flood_fill_button.clicked.connect(self.floodFillRequested.emit)
        flood_fill_layout.addWidget(self.flood_fill_button)
        
        # Instructions
        instructions = QLabel("Click on a region in the image to flood fill it with the selected color.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        flood_fill_layout.addWidget(instructions)
        
        layout.addWidget(flood_fill_group)
        
        # Region Smoothing Section
        smoothing_group = QGroupBox("Region Smoothing")
        smoothing_layout = QFormLayout()
        
        # Smoothing method dropdown
        self.smoothing_method_combo = QComboBox()
        self.smoothing_method_combo.addItems([
            "None",
            "Morphological",
            "Bilateral",
            "Contour",
            "Gaussian",
            "Multi-scale"
        ])
        self.smoothing_method_combo.setCurrentText("None")
        smoothing_layout.addRow("Method:", self.smoothing_method_combo)
        
        # Smoothing strength slider
        self.smoothing_strength_slider = QSlider(Qt.Horizontal)
        self.smoothing_strength_slider.setRange(0, 100)
        self.smoothing_strength_slider.setValue(50)
        self.smoothing_strength_slider.valueChanged.connect(self._on_smoothing_strength_changed)
        
        self.smoothing_strength_label = QLabel("50%")
        smoothing_strength_layout = QHBoxLayout()
        smoothing_strength_layout.addWidget(self.smoothing_strength_slider)
        smoothing_strength_layout.addWidget(self.smoothing_strength_label)
        smoothing_layout.addRow("Strength:", smoothing_strength_layout)
        
        # Preserve colors checkbox
        self.preserve_colors_checkbox = QCheckBox("Preserve original colors")
        self.preserve_colors_checkbox.setChecked(True)
        smoothing_layout.addRow("", self.preserve_colors_checkbox)
        
        # Smoothing button
        self.smooth_button = QPushButton("Apply Smoothing")
        self.smooth_button.clicked.connect(self._on_smooth_requested)
        smoothing_layout.addRow("", self.smooth_button)
        
        smoothing_group.setLayout(smoothing_layout)
        layout.addWidget(smoothing_group)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        self.cleanup_button = QPushButton("Clean Up Regions")
        self.cleanup_button.clicked.connect(self.cleanupRequested.emit)
        actions_layout.addWidget(self.cleanup_button)
        
        self.preview_checkbox = QCheckBox("Show Preview")
        self.preview_checkbox.toggled.connect(self.previewToggled.emit)
        actions_layout.addWidget(self.preview_checkbox)
        
        self.apply_button = QPushButton("Apply Cleanup")
        self.apply_button.clicked.connect(self.applyRequested.emit)
        actions_layout.addWidget(self.apply_button)
        
        layout.addLayout(actions_layout)
        
        # Save button
        save_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Working Image")
        self.save_button.clicked.connect(self._on_save_requested)
        save_layout.addWidget(self.save_button)
        save_layout.addStretch(1)  # Push button to the left
        layout.addLayout(save_layout)
        
        layout.addStretch(1)
    
    def _on_size_threshold_changed(self, value: int) -> None:
        """Handle size threshold slider change."""
        self._min_region_size = value
        self.size_label.setText(f"{value} pixels")
    
    def _on_auto_merge_threshold_changed(self, value: int) -> None:
        """Handle auto-merge threshold slider change."""
        self.auto_merge_label.setText(f"{value}%")
    
    def _on_overlay_toggled(self, checked: bool) -> None:
        """Handle region overlay checkbox toggle."""
        self._show_region_overlay = checked
        self.regionBoundariesToggled.emit(checked)
    
    def get_min_region_size(self) -> int:
        """Get the minimum region size threshold."""
        return self._min_region_size
    
    def get_auto_merge_threshold(self) -> float:
        """Get the auto-merge threshold as a float between 0 and 1."""
        return self.auto_merge_threshold_slider.value() / 100.0
    
    def get_show_region_overlay(self) -> bool:
        """Get whether to show region overlay."""
        return self._show_region_overlay
    
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
    
    def update_flood_fill_palette(self, colors: List[QColor]) -> None:
        """Update the flood fill color palette."""
        # Clear existing palette
        for i in reversed(range(self.palette_layout.count())):
            child = self.palette_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Add new color swatches
        cols = 6
        for i, color in enumerate(colors):
            swatch = ColorSwatch(color)
            swatch.colorSelected.connect(self._on_flood_fill_color_selected)
            row = i // cols
            col = i % cols
            self.palette_layout.addWidget(swatch, row, col)
        
        # Set first color as default selection if available
        if colors:
            self._on_flood_fill_color_selected(colors[0])
    
    def _on_flood_fill_color_selected(self, color: QColor) -> None:
        """Handle flood fill color selection."""
        self._selected_flood_fill_color = color
        self.selected_color_swatch.color = color
        self.selected_color_swatch.update_display()
    
    def get_selected_flood_fill_color(self) -> Optional[QColor]:
        """Get the currently selected flood fill color."""
        return getattr(self, '_selected_flood_fill_color', None)
    
    def _on_smoothing_strength_changed(self, value: int) -> None:
        """Handle smoothing strength slider change."""
        self.smoothing_strength_label.setText(f"{value}%")
    
    def _on_smooth_requested(self) -> None:
        """Handle smoothing button click."""
        method = self.smoothing_method_combo.currentText().lower()
        if method == "none":
            return
        
        # Convert method names to internal format
        method_map = {
            "morphological": "morphological",
            "bilateral": "bilateral", 
            "contour": "contour",
            "gaussian": "gaussian",
            "multi-scale": "multiscale"
        }
        
        internal_method = method_map.get(method, "morphological")
        strength = self.smoothing_strength_slider.value() / 100.0
        preserve_colors = self.preserve_colors_checkbox.isChecked()
        
        self.smoothingRequested.emit(internal_method, strength, preserve_colors)
    
    def _on_save_requested(self) -> None:
        """Handle save button click."""
        self.saveRequested.emit()
