from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, 
    QGroupBox, QFormLayout, QCheckBox, QSpinBox, QComboBox, QScrollArea,
    QGridLayout, QFrame, QMessageBox, QColorDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QApplication
)

from model import AppState
from processing.color_simplify import create_palette_from_colors, get_color_statistics, simplify_colors_adaptive
from ui.algorithm_preview_dialog import AlgorithmPreviewDialog
from ui.base_step import BaseStep
from ui.image_view import ImageView
from ui.color_processing.eyedropper_view import EyedropperView


class ColorSwatchWidget(QLabel):
    """A widget that displays a colored rectangle as a color swatch."""
    
    def __init__(self, color: QColor, hsize: int = 40, vsize: int = 20, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(hsize, vsize)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)
        self.update_display()
    
    def update_display(self):
        """Update the visual display of the color swatch."""
        # Create a pixmap with the color
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        
        # Fill with the color
        painter.fillRect(pixmap.rect(), self.color)
        
        # Draw border
        painter.setPen(QPen(Qt.black, 1))
        painter.drawRect(pixmap.rect().adjusted(0, 0, -1, -1))
        
        painter.end()
        
        # Set the pixmap as the label's pixmap
        self.setPixmap(pixmap)
    
    def set_color(self, color: QColor):
        """Update the color of the swatch."""
        self.color = color
        self.update_display()

class ColorProcessingPanel(BaseStep):
    statusBarMessage = Signal(str)
    """Combined panel for color simplification and custom palette operations."""
    
    def __init__(self, parent: Optional[QWidget] = None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None) -> None:
        super().__init__(parent, app_state, image_view)
        self._colors: List[QColor] = []
        self._distance_metric = "lab"
        self._eyedropper_active = False
        
        # Create the eyedropper view
        self._eyedropper_view = EyedropperView(self._image_view, self._app_state, self._image_view)
        
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        # Create the main content widget
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
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
        self.combo_algorithm.addItem("Threshold/Posterize (Simple)", "threshold")
        self.combo_algorithm.addItem("Perceptual Clustering (Smart Similar)", "perceptual")
        self.combo_algorithm.addItem("Perceptual Clustering (Fast)", "perceptual_fast")
        self.combo_algorithm.addItem("HSV Clustering (Color Families)", "hsv_clustering")
        self.combo_algorithm.addItem("Adaptive (Auto-select)", "adaptive")
        color_form.addRow("Algorithm", self.combo_algorithm)
        
        layout.addWidget(color_group)
        
        # Custom Palette Section
        palette_group = QGroupBox("Custom Palette")
        palette_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        palette_layout = QVBoxLayout(palette_group)
        
        # Palette instructions
        palette_instructions = QLabel(
            "Create a custom palette by adding colors manually or using the eyedropper to pick colors from the image."
        )
        palette_instructions.setWordWrap(True)
        palette_layout.addWidget(palette_instructions)
        
        # Palette controls (moved above the table)
        palette_controls = QHBoxLayout()
        
        self.btn_add_color = QPushButton("Add Color")
        palette_controls.addWidget(self.btn_add_color)
        
        self.btn_eyedropper = QPushButton("Eyedropper")
        palette_controls.addWidget(self.btn_eyedropper)
        
        self.btn_remove_color = QPushButton("Remove Color")
        self.btn_remove_color.setEnabled(False)  # Initially disabled
        palette_controls.addWidget(self.btn_remove_color)
        
        self.btn_clear = QPushButton("Clear Palette")
        palette_controls.addWidget(self.btn_clear)
        
        palette_layout.addLayout(palette_controls)
        
        
        # Create a scrollable area for the palette table
        self.palette_scroll = QScrollArea()
        self.palette_scroll.setWidgetResizable(True)
        self.palette_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.palette_table = QTableWidget()
        self.palette_table.setColumnCount(2)
        self.palette_table.setHorizontalHeaderLabels(["Color", "RGB"])
        self.palette_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.palette_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.palette_table.setColumnWidth(0, 50)  # Fixed width for color swatch
        self.palette_table.setAlternatingRowColors(True)
        
        # Set custom styling for alternating rows and selection
        self.palette_table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #606060;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #2c5aa0;
            }
        """)
        
        # Set selection behavior to select entire rows
        self.palette_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.palette_table.setSelectionMode(QTableWidget.SingleSelection)
        self.palette_table.setFocusPolicy(Qt.StrongFocus)  # Ensure table can receive focus
        
        
        self.palette_scroll.setWidget(self.palette_table)
        palette_layout.addWidget(self.palette_scroll)
        
        # Distance metric and process button (moved below the table)
        bottom_controls = QHBoxLayout()
        
        # Distance metric selection
        bottom_controls.addWidget(QLabel("Distance Metric:"))
        
        self.combo_metric = QComboBox()
        self.combo_metric.addItem("LAB (Perceptual)", "lab")
        self.combo_metric.addItem("RGB (Euclidean)", "rgb")
        self.combo_metric.addItem("HSV", "hsv")
        bottom_controls.addWidget(self.combo_metric)
        
        bottom_controls.addStretch()
        
        # Process button
        self.btn_palette_process = QPushButton("Process with Custom Palette")
        bottom_controls.addWidget(self.btn_palette_process)
        
        palette_layout.addLayout(bottom_controls)
        
        layout.addWidget(palette_group)
        
        
        # Algorithm actions
        algorithm_actions = QHBoxLayout()
        self.btn_algorithm_process = QPushButton("Process with Algorithm")
        self.btn_preview_all = QPushButton("Preview All Outputs")
        algorithm_actions.addWidget(self.btn_algorithm_process)
        algorithm_actions.addWidget(self.btn_preview_all)
        color_form.addRow("", algorithm_actions)
        
        # Set the main widget
        self.set_main_widget(main_widget)

    def _get_algorithm_name(self, algorithm: str) -> str:
        """Get the name of the algorithm."""
        return {
            "kmeans": "K-means Clustering",
            "median_cut": "Median Cut",
            "octree": "Octree Quantization", 
            "threshold": "Threshold/Posterize",
            "perceptual": "Perceptual Clustering",
            "perceptual_fast": "Perceptual Clustering (Fast)",
            "adaptive_distance": "Adaptive Distance",
            "hsv_clustering": "HSV Clustering",
            "adaptive": "Adaptive"
        }.get(algorithm, algorithm)
    
    def _on_algorithm_process(self) -> None:
        """Handle algorithm-based color processing request."""
        # Always use the working image as the source for processing
        base_image = self._app_state.base_image
        if base_image is None:
            return
        source_rgba = base_image.copy()
        num_colors = self.get_num_colors()
        algorithm = self.get_algorithm()
        algorithm_name = self._get_algorithm_name(algorithm)
        
        # Set wait cursor and show status
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.statusBarMessage.emit(f"Processing colors with {algorithm_name}...")
            QApplication.processEvents()  # Ensure status message is displayed
            
            simplified_rgba, palette = simplify_colors_adaptive(source_rgba, num_colors, algorithm)
            self.set_working_image(simplified_rgba.copy())
            stats = get_color_statistics(simplified_rgba)
            stats['palette'] = palette
            self.update_statistics(stats)
            self.statusBarMessage.emit(f"Color processing complete using {algorithm_name}. Reduced to {num_colors} colors.")
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()

    def _on_preview_all(self) -> None:
        """Handle preview all outputs request."""
        base_image = self._app_state.base_image
        if base_image is None:
            return
        source_rgba = base_image.copy()
        num_colors = self.get_num_colors()
		
		# Create and show preview dialog
        dialog = AlgorithmPreviewDialog(source_rgba, num_colors)
        dialog.algorithm_selected.connect(self._on_algorithm_selected_from_preview)
        dialog.exec()

    def _on_algorithm_selected_from_preview(self, algorithm_name: str, result_rgba: np.ndarray, palette: np.ndarray) -> None:
        """Handle algorithm selection from preview dialog."""
        # Set the working image
        self.set_working_image(result_rgba.copy())
		# Update statistics
        stats = {
			'algorithm': algorithm_name.lower().replace(' ', '_').replace('(', '').replace(')', ''),
			'num_colors': len(palette),
			'palette': palette,
			'processing_time': 0.0
		}
        self.update_statistics(stats)
    
    def _on_palette_process(self) -> None:
        """Handle custom palette processing request."""
        # Always use the base image as the source for processing
        base_image = self._app_state.base_image
        if base_image is None:
            return
        source_rgba = base_image.copy()
        
        # Get parameters from the color processing panel
        custom_palette = self.get_palette()
        distance_metric = self.get_distance_metric()
        
        # Set wait cursor and show status
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.statusBarMessage.emit("Processing colors with custom palette...")
            QApplication.processEvents()  # Ensure status message is displayed
            
            # Perform custom palette simplification
            from processing.color_simplify import simplify_colors_custom_palette
            simplified_rgba, palette = simplify_colors_custom_palette(
                source_rgba, 
                custom_palette, 
                distance_metric
            )
            
            # Store the simplified output
            self.set_working_image(simplified_rgba.copy())
            
            # Update statistics with palette information
            stats = get_color_statistics(simplified_rgba)
            stats['palette'] = palette
            self.update_statistics(stats)
            
            metric_name = {
                "lab": "LAB (Perceptual)",
                "rgb": "RGB (Euclidean)",
                "hsv": "HSV"
            }.get(distance_metric, distance_metric)
            
            self.statusBarMessage.emit(f"Custom palette processing complete using {metric_name}. Applied {len(palette)} colors.")
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
    
    def _add_color(self) -> None:
        """Add a new color to the palette."""
        color = QColorDialog.getColor(Qt.white, self)
        if color.isValid():
            self._add_color_to_list(color)
    
    def _add_color_to_list(self, color: QColor) -> None:
        """Add a color to the internal list."""
        self._colors.append(color)
        self._update_stats()
        self._update_buttons()
    
    def _clear_palette(self) -> None:
        """Clear all colors from the palette."""
        self._colors.clear()
        self._update_stats()
        self._update_buttons()
    
    def _on_table_selection_changed(self) -> None:
        """Handle table selection changes to enable/disable Remove Color button."""
        selected_rows = self.palette_table.selectionModel().selectedRows()
        has_selection = len(selected_rows) > 0
        self.btn_remove_color.setEnabled(has_selection)
    
    def _check_selection_state(self) -> None:
        """Manually check and update the Remove Color button state."""
        self._on_table_selection_changed()
    
    def _remove_selected_color(self) -> None:
        """Remove the selected color from the palette."""
        selected_rows = self.palette_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        # Get the row index of the first selected row
        row = selected_rows[0].row()
        
        # Remove the color from the internal list
        if 0 <= row < len(self._colors):
            del self._colors[row]
            self._update_stats()
            
            # Select the next row (or previous if we removed the last row)
            new_row_count = len(self._colors)
            if new_row_count > 0:
                # Select the same index, or the last row if we removed the last item
                next_row = min(row, new_row_count - 1)
                self.palette_table.selectRow(next_row)
            else:
                # No more colors, clear selection
                self.palette_table.clearSelection()
            
            self._update_buttons()
    
    def _start_eyedropper(self) -> None:
        """Toggle the eyedropper tool."""
        if self._eyedropper_active:
            # Deactivate eyedropper
            self._eyedropper_active = False
            self._eyedropper_view.set_active(False)
            self.reset_eyedropper_state()
        else:
            # Activate eyedropper
            self._eyedropper_active = True
            self._eyedropper_view.set_active(True)
            # Update button appearance to show it's active
            self.btn_eyedropper.setText("Eyedropper (Active)")
            self.btn_eyedropper.setStyleSheet("background-color: #4CAF50; color: white;")
    
    def reset_eyedropper_state(self) -> None:
        """Reset the eyedropper button to its default state."""
        self._eyedropper_active = False
        self.btn_eyedropper.setText("Eyedropper")
        self.btn_eyedropper.setStyleSheet("")
    
    def _update_stats(self) -> None:
        """Update the palette table display."""
        count = len(self._colors)
        if count == 0:
            self.palette_table.setRowCount(0)
        else:
            # Update the palette table with current colors
            self.palette_table.setRowCount(count)
            for i, color in enumerate(self._colors):
                # Create color swatch widget
                color_swatch = ColorSwatchWidget(color)
                self.palette_table.setCellWidget(i, 0, color_swatch)
                
                # Create RGB text item
                rgb_text = f"RGB({color.red()}, {color.green()}, {color.blue()})"
                rgb_item = QTableWidgetItem(rgb_text)
                rgb_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # Enable selection but not editing
                self.palette_table.setItem(i, 1, rgb_item)
            
            # Resize rows to fit content
            self.palette_table.resizeRowsToContents()
            
            # Check selection state after updating the table
            self._check_selection_state()
    
    def _update_buttons(self) -> None:
        """Update button states based on palette content."""
        has_colors = len(self._colors) > 0
        self.btn_algorithm_process.setEnabled(True)  # Always enabled for algorithm-based processing
        self.btn_palette_process.setEnabled(has_colors)  # Only enabled if there are colors in palette
        
        # Update Remove Color button state based on table selection
        self._on_table_selection_changed()
    
    def get_num_colors(self) -> int:
        """Get the target number of colors."""
        return self.spin_num_colors.value()
    
    def get_algorithm(self) -> str:
        """Get the selected algorithm."""
        return self.combo_algorithm.currentData()
    
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
        """Update palette table with color information from algorithm processing."""
        if not stats or 'palette' not in stats or stats['palette'] is None:
            # If no algorithm results, clear the palette
            self._colors.clear()
            self._update_stats()
            return
        
        # Clear existing colors and add only algorithm results
        self._colors.clear()
        palette = stats['palette']
        for color in palette:
            qcolor = QColor(int(color[0]), int(color[1]), int(color[2]))
            self._colors.append(qcolor)
        
        # Update the display with algorithm colors only
        self._update_stats()
        
        # Resize rows to fit content
        self.palette_table.resizeRowsToContents()
    
    def is_eyedropper_active(self) -> bool:
        """Check if eyedropper is active."""
        return self._eyedropper_active
        
    def _on_open(self) -> None:
        """Handle the panel being opened."""
        super()._on_open()
        self.btn_add_color.clicked.connect(self._add_color)
        self.btn_eyedropper.clicked.connect(self._start_eyedropper)
        self.btn_remove_color.clicked.connect(self._remove_selected_color)
        self.btn_clear.clicked.connect(self._clear_palette)
        self.btn_algorithm_process.clicked.connect(self._on_algorithm_process)
        self.btn_preview_all.clicked.connect(self._on_preview_all)
        self.btn_palette_process.clicked.connect(self._on_palette_process)
        
        # Connect eyedropper view signals
        self._eyedropper_view.color_picked.connect(self.add_color_from_image)
        self._eyedropper_view.eyedropper_cancelled.connect(self.reset_eyedropper_state)
        
        self.palette_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.palette_table.selectionModel().selectionChanged.connect(self._on_table_selection_changed)


    def _on_close(self) -> None:
        """Handle the panel being closed."""
        super()._on_close()
        self.btn_add_color.clicked.disconnect(self._add_color)
        self.btn_eyedropper.clicked.disconnect(self._start_eyedropper)
        self.btn_remove_color.clicked.disconnect(self._remove_selected_color)
        self.btn_clear.clicked.disconnect(self._clear_palette)
        self.btn_algorithm_process.clicked.disconnect(self._on_algorithm_process)
        self.btn_preview_all.clicked.disconnect(self._on_preview_all)
        self.btn_palette_process.clicked.disconnect(self._on_palette_process)

        # Disconnect eyedropper view signals
        self._eyedropper_view.color_picked.disconnect(self.add_color_from_image)
        self._eyedropper_view.eyedropper_cancelled.disconnect(self.reset_eyedropper_state)
        
        # Deactivate eyedropper if active
        if self._eyedropper_active:
            self._eyedropper_view.set_active(False)
            self.reset_eyedropper_state()

        self.palette_table.itemSelectionChanged.disconnect(self._on_table_selection_changed)
        self.palette_table.selectionModel().selectionChanged.disconnect(self._on_table_selection_changed)