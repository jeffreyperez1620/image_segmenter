from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, 
    QGroupBox, QFormLayout, QCheckBox, QSpinBox, QComboBox, QScrollArea,
    QGridLayout, QFrame, QMessageBox, QColorDialog
)

from processing.color_simplify import create_palette_from_colors


class ColorSwatch(QLabel):
    """A color swatch widget that can be clicked to edit or remove colors."""
    
    colorChanged = Signal(int, QColor)  # index, new_color
    colorRemoved = Signal(int)  # index
    colorSelected = Signal(QColor)  # for flood fill selection
    
    def __init__(self, color: QColor, index: int, parent: Optional[QWidget] = None, selectable: bool = False) -> None:
        super().__init__(parent)
        self.color = color
        self.index = index
        self.selectable = selectable
        self.setFixedSize(40, 40)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        self.setMidLineWidth(1)
        self.update_display()
    
    def update_display(self) -> None:
        """Update the visual display of the color swatch."""
        # Create a pixmap with the color
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        
        # Fill with the color
        painter.fillRect(pixmap.rect(), self.color)
        
        # Draw border
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(pixmap.rect().adjusted(1, 1, -1, -1))
        
        painter.end()
        
        # Set the pixmap as the label's pixmap
        self.setPixmap(pixmap)
    
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """Handle mouse clicks to edit, remove, or select colors."""
        if event.button() == Qt.LeftButton:
            if self.selectable:
                # For flood fill selection
                self.colorSelected.emit(self.color)
            else:
                # Left click to edit color
                new_color = QColorDialog.getColor(self.color, self)
                if new_color.isValid() and new_color != self.color:
                    self.color = new_color
                    self.update_display()
                    self.colorChanged.emit(self.index, new_color)
        elif event.button() == Qt.RightButton and not self.selectable:
            # Right click to remove color
            self.colorRemoved.emit(self.index)


class ColorProcessingPanel(QWidget):
    """Combined panel for color simplification and custom palette operations."""
    
    algorithmProcessRequested = Signal()
    paletteProcessRequested = Signal()
    previewToggled = Signal(bool)
    applyRequested = Signal()
    eyedropperRequested = Signal()
    eyedropperCancelled = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._colors: List[QColor] = []
        self._distance_metric = "lab"
        self._preserve_alpha = True
        self._eyedropper_active = False
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Color Processing")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Simplify colors using algorithms or create a custom palette. Use the eyedropper to pick colors from the image, "
            "or use flood-fill to manually adjust regions."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Color count control
        color_group = QGroupBox("Color Reduction", self)
        color_form = QFormLayout(color_group)
        
        self.spin_num_colors = QSpinBox()
        self.spin_num_colors.setMinimum(2)
        self.spin_num_colors.setMaximum(256)
        self.spin_num_colors.setValue(8)
        color_form.addRow("Number of Colors", self.spin_num_colors)
        
        # Algorithm selection
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItem("K-means Clustering (Best Quality)", "kmeans")
        self.combo_algorithm.addItem("Median Cut (Balanced)", "median_cut")
        self.combo_algorithm.addItem("Octree Quantization (Fast)", "octree")
        self.combo_algorithm.addItem("Threshold/Posterize (Simple)", "threshold")
        self.combo_algorithm.addItem("Perceptual Clustering (Smart Similar)", "perceptual")
        self.combo_algorithm.addItem("Perceptual Clustering (Fast)", "perceptual_fast")
        self.combo_algorithm.addItem("Adaptive Distance (Similar Shades)", "adaptive_distance")
        self.combo_algorithm.addItem("HSV Clustering (Color Families)", "hsv_clustering")
        self.combo_algorithm.addItem("Adaptive (Auto-select)", "adaptive")
        self.combo_algorithm.addItem("Custom Palette", "custom_palette")
        color_form.addRow("Algorithm", self.combo_algorithm)
        
        # Alpha preservation
        self.chk_preserve_alpha = QCheckBox("Preserve Alpha Channel")
        self.chk_preserve_alpha.setChecked(True)
        color_form.addRow("", self.chk_preserve_alpha)
        
        layout.addWidget(color_group)
        
        # Custom Palette Section
        palette_group = QGroupBox("Custom Palette")
        palette_layout = QVBoxLayout(palette_group)
        
        # Palette instructions
        palette_instructions = QLabel(
            "Create a custom palette by adding colors manually or using the eyedropper to pick colors from the image."
        )
        palette_instructions.setWordWrap(True)
        palette_layout.addWidget(palette_instructions)
        
        # Palette controls
        palette_controls = QHBoxLayout()
        
        self.btn_add_color = QPushButton("Add Color")
        self.btn_add_color.clicked.connect(self._add_color)
        palette_controls.addWidget(self.btn_add_color)
        
        self.btn_eyedropper = QPushButton("Eyedropper")
        self.btn_eyedropper.clicked.connect(self._start_eyedropper)
        palette_controls.addWidget(self.btn_eyedropper)
        
        self.btn_clear = QPushButton("Clear Palette")
        self.btn_clear.clicked.connect(self._clear_palette)
        palette_controls.addWidget(self.btn_clear)
        
        palette_layout.addLayout(palette_controls)
        
        # Color swatches area
        self.swatches_layout = QGridLayout()
        self.swatches_widget = QWidget()
        self.swatches_widget.setLayout(self.swatches_layout)
        
        # Scroll area for swatches
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.swatches_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(120)
        palette_layout.addWidget(scroll_area)
        
        # Distance metric selection
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Distance Metric:"))
        
        self.combo_metric = QComboBox()
        self.combo_metric.addItem("LAB (Perceptual)", "lab")
        self.combo_metric.addItem("RGB (Euclidean)", "rgb")
        self.combo_metric.addItem("HSV", "hsv")
        metric_layout.addWidget(self.combo_metric)
        
        palette_layout.addLayout(metric_layout)
        
        # Custom palette actions
        palette_actions = QHBoxLayout()
        self.btn_palette_process = QPushButton("Process with Custom Palette")
        palette_actions.addWidget(self.btn_palette_process)
        palette_layout.addLayout(palette_actions)
        
        layout.addWidget(palette_group)
        
        # Statistics display
        stats_group = QGroupBox("Color Statistics", self)
        stats_layout = QVBoxLayout(stats_group)
        self.label_stats = QLabel("No image loaded")
        self.label_stats.setWordWrap(True)
        stats_layout.addWidget(self.label_stats)
        layout.addWidget(stats_group)
        
        
        # Algorithm actions
        algorithm_actions = QHBoxLayout()
        self.btn_algorithm_process = QPushButton("Process with Algorithm")
        self.chk_preview = QCheckBox("Show Preview")
        algorithm_actions.addWidget(self.btn_algorithm_process)
        algorithm_actions.addWidget(self.chk_preview)
        color_form.addRow("", algorithm_actions)
        
        # Apply button
        self.btn_apply = QPushButton("Apply Processing")
        layout.addWidget(self.btn_apply)
        
        layout.addStretch(1)
        
        # Wire signals
        self.btn_algorithm_process.clicked.connect(self._on_algorithm_process)
        self.btn_palette_process.clicked.connect(self._on_palette_process)
        self.chk_preview.toggled.connect(self.previewToggled)
        self.btn_apply.clicked.connect(self.applyRequested)
    
    def _on_algorithm_process(self) -> None:
        """Handle algorithm-based processing request."""
        self.algorithmProcessRequested.emit()
    
    def _on_palette_process(self) -> None:
        """Handle custom palette processing request."""
        self.paletteProcessRequested.emit()
    
    def _add_color(self) -> None:
        """Add a new color to the palette."""
        color = QColorDialog.getColor(Qt.white, self)
        if color.isValid():
            self._add_color_to_list(color)
    
    def _add_color_to_list(self, color: QColor) -> None:
        """Add a color to the internal list and create a swatch."""
        self._colors.append(color)
        self._update_swatches()
        self._update_stats()
        self._update_buttons()
    
    def _update_swatches(self) -> None:
        """Update the color swatches display."""
        # Clear existing swatches
        for i in reversed(range(self.swatches_layout.count())):
            child = self.swatches_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add new swatches
        cols = 8  # Number of columns
        for i, color in enumerate(self._colors):
            swatch = ColorSwatch(color, i)
            swatch.colorChanged.connect(self._on_color_changed)
            swatch.colorRemoved.connect(self._on_color_removed)
            
            row = i // cols
            col = i % cols
            self.swatches_layout.addWidget(swatch, row, col)
    
    def _on_color_changed(self, index: int, new_color: QColor) -> None:
        """Handle color change in a swatch."""
        if 0 <= index < len(self._colors):
            self._colors[index] = new_color
            self._update_stats()
    
    def _on_color_removed(self, index: int) -> None:
        """Handle color removal from a swatch."""
        if 0 <= index < len(self._colors):
            del self._colors[index]
            self._update_swatches()
            self._update_stats()
            self._update_buttons()
    
    def _clear_palette(self) -> None:
        """Clear all colors from the palette."""
        self._colors.clear()
        self._update_swatches()
        self._update_stats()
        self._update_buttons()
    
    def _start_eyedropper(self) -> None:
        """Toggle the eyedropper tool."""
        if self._eyedropper_active:
            # Deactivate eyedropper
            self._eyedropper_active = False
            self.eyedropperCancelled.emit()
            self.reset_eyedropper_state()
        else:
            # Activate eyedropper
            self._eyedropper_active = True
            self.eyedropperRequested.emit()
            # Update button appearance to show it's active
            self.btn_eyedropper.setText("Eyedropper (Active)")
            self.btn_eyedropper.setStyleSheet("background-color: #4CAF50; color: white;")
    
    def reset_eyedropper_state(self) -> None:
        """Reset the eyedropper button to its default state."""
        self._eyedropper_active = False
        self.btn_eyedropper.setText("Eyedropper")
        self.btn_eyedropper.setStyleSheet("")
    
    def _update_stats(self) -> None:
        """Update the statistics display."""
        count = len(self._colors)
        if count == 0:
            self.label_stats.setText("No colors in palette")
        else:
            self.label_stats.setText(f"Palette: {count} colors")
    
    def _update_buttons(self) -> None:
        """Update button states based on palette content."""
        has_colors = len(self._colors) > 0
        self.btn_algorithm_process.setEnabled(True)  # Always enabled for algorithm-based processing
        self.btn_palette_process.setEnabled(has_colors)  # Only enabled if there are colors in palette
        self.btn_apply.setEnabled(True)     # Always enabled
    
    def get_num_colors(self) -> int:
        """Get the target number of colors."""
        return self.spin_num_colors.value()
    
    def get_algorithm(self) -> str:
        """Get the selected algorithm."""
        return self.combo_algorithm.currentData()
    
    def get_preserve_alpha(self) -> bool:
        """Get whether to preserve alpha channel."""
        return self.chk_preserve_alpha.isChecked()
    
    def get_palette(self) -> np.ndarray:
        """Get the current palette as a numpy array."""
        if not self._colors:
            return np.array([[0, 0, 0]], dtype=np.uint8)
        
        colors_list = [(color.red(), color.green(), color.blue()) for color in self._colors]
        return create_palette_from_colors(colors_list)
    
    def get_distance_metric(self) -> str:
        """Get the current distance metric."""
        return self.combo_metric.currentData()
    
    def add_color_from_image(self, color: QColor) -> None:
        """Add a color picked from the image."""
        self._add_color_to_list(color)
        # Keep eyedropper button active for multiple picks
        # Don't reset the button appearance
    
    def update_statistics(self, stats: dict) -> None:
        """Update statistics display with image information."""
        if not stats:
            self.label_stats.setText("No image loaded")
            return
        
        text = f"""
        Image Size: {stats.get('image_size', (0, 0))[0]} × {stats.get('image_size', (0, 0))[1]}
        Unique Colors: {stats.get('total_unique_colors', 0):,}
        Non-transparent Pixels: {stats.get('non_transparent_pixels', 0):,}
        RGB Mean: ({stats.get('rgb_mean', [0,0,0])[0]:.1f}, {stats.get('rgb_mean', [0,0,0])[1]:.1f}, {stats.get('rgb_mean', [0,0,0])[2]:.1f})
        RGB Std: ({stats.get('rgb_std', [0,0,0])[0]:.1f}, {stats.get('rgb_std', [0,0,0])[1]:.1f}, {stats.get('rgb_std', [0,0,0])[2]:.1f})
        """.strip()
        
        # Add palette information if available
        if 'palette' in stats and stats['palette'] is not None:
            palette = stats['palette']
            text += f"\n\nPalette ({len(palette)} colors):"
            for i, color in enumerate(palette[:8]):  # Show first 8 colors
                text += f"\n  {i+1}: RGB({color[0]}, {color[1]}, {color[2]})"
            if len(palette) > 8:
                text += f"\n  ... and {len(palette) - 8} more colors"
        
        self.label_stats.setText(text)
    
    def is_eyedropper_active(self) -> bool:
        """Check if eyedropper is active."""
        return self._eyedropper_active
    
    
    def update_color_preview(self, color: QColor) -> None:
        """Update the color preview display with the given color."""
        # This method is called by the image view when hovering over colors
        # For now, we don't have a preview display in the combined panel
        # This could be added later if needed
        pass
