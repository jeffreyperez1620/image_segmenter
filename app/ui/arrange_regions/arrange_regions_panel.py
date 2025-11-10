"""
Panel for arranging regions by color.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSizePolicy
)

from model import AppState
from ui.base_step import BaseStep
from ui.image_view import ImageView
from processing.arrange_regions import extract_regions_by_color, ColorRegions, RegionData
from utils.qt_image import numpy_rgba_to_qimage
from ui.arrange_regions.region_overlay_view import RegionOverlayView


class ArrangeRegionsPanel(BaseStep):
    """Panel for arranging regions by color."""

    statusBarMessage = Signal(str)
    
    def __init__(self, parent: Optional[QWidget] = None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None):
        super().__init__(parent, app_state, image_view)
        
        # Region model: Dict[color_tuple, ColorRegions]
        self._region_model: Dict[Tuple[int, int, int], ColorRegions] = {}
        
        # Currently selected color
        self._selected_color: Optional[Tuple[int, int, int]] = None
        
        # Region overlay view
        self._region_overlay_view: Optional[RegionOverlayView] = None
        
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        # Create main widget
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setAlignment(Qt.AlignTop)
        
        # Create region overlay view
        self._region_overlay_view = RegionOverlayView(self._image_view, self._app_state, self._image_view)
        
        # Instructions
        instructions = QLabel(
            "Select a color from the palette to view and arrange its regions. "
            "Drag regions to reposition them. Use Reset to restore original positions."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Color Palette section
        palette_group = QGroupBox("Color Palette")
        palette_layout = QVBoxLayout(palette_group)
        
        # Create palette table
        self.palette_table = QTableWidget()
        self.palette_table.setColumnCount(2)
        self.palette_table.setHorizontalHeaderLabels(["Color", "RGB"])
        self.palette_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.palette_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.palette_table.setColumnWidth(0, 50)
        self.palette_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.palette_table.setSelectionMode(QTableWidget.SingleSelection)
        
        palette_layout.addWidget(self.palette_table)
        layout.addWidget(palette_group)
        
        # Add stretch at the end
        layout.addStretch(1)
        
        # Set the main widget
        self.set_main_widget(main_widget)
    
    def _build_region_model(self) -> None:
        """Build the region model from the base image."""
        base_image = self._app_state.base_image
        if base_image is None:
            self._region_model = {}
            return
        
        # Extract regions by color
        self._region_model = extract_regions_by_color(base_image, connectivity=8)
    
    def _update_palette_table(self) -> None:
        """Update the palette table with colors from the region model."""
        self.palette_table.setRowCount(0)
        
        if not self._region_model:
            return
        
        # Sort colors for consistent display
        colors = sorted(self._region_model.keys())
        
        self.palette_table.setRowCount(len(colors))
        
        for i, color_tuple in enumerate(colors):
            color = QColor(color_tuple[0], color_tuple[1], color_tuple[2])
            color_regions = self._region_model[color_tuple]
            
            # Create color swatch
            from ui.color_processing.color_processing_panel import ColorSwatchWidget
            color_swatch = ColorSwatchWidget(color)
            self.palette_table.setCellWidget(i, 0, color_swatch)
            
            # Create RGB text item
            rgb_text = f"RGB({color_tuple[0]}, {color_tuple[1]}, {color_tuple[2]}) - {len(color_regions.regions)} region(s)"
            rgb_item = QTableWidgetItem(rgb_text)
            rgb_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.palette_table.setItem(i, 1, rgb_item)
    
    def _on_color_selected(self) -> None:
        """Handle color selection from the palette table."""
        selected_rows = self.palette_table.selectionModel().selectedRows()
        
        if not selected_rows:
            # No selection - show base image
            self._selected_color = None
            self._update_region_display()
            return
        
        row = selected_rows[0].row()
        colors = sorted(self._region_model.keys())
        
        if 0 <= row < len(colors):
            self._selected_color = colors[row]
            color_regions = self._region_model.get(self._selected_color)
            if color_regions:
                self.statusBarMessage.emit(f"Selected color with {len(color_regions.regions)} region(s)")
            self._update_region_display()
    
    def _update_region_display(self) -> None:
        """Update the region display based on selected color."""
        base_image = self._app_state.base_image
        if base_image is None:
            return
        
        if self._selected_color is None:
            # No color selected - always show base image
            self._region_overlay_view.set_regions(None)
            self._image_view.set_show_working(False)  # Show base image
        else:
            # Color selected - show regions of that color
            color_regions = self._region_model.get(self._selected_color)
            if color_regions:
                # Set and display regions
                self._region_overlay_view.set_regions(color_regions)
                
                # Set background based on checkbox
                # show_base = True means show base image (set_show_working(False))
                # show_base = False means show working image (transparent, set_show_working(True))
                show_base = self._chk_show_base.isChecked()
                self._image_view.set_show_working(not show_base)
    
    def _on_reset_clicked(self) -> None:
        """Override Reset button to restore original region positions."""
        if not self._region_model:
            return
        
        # Restore original positions for all regions
        for color_regions in self._region_model.values():
            for region in color_regions.regions:
                region.position = region.original_position
        
        # Update display if a color is selected
        if self._selected_color and self._region_overlay_view:
            # Refresh the overlay to show updated positions
            color_regions = self._region_model.get(self._selected_color)
            if color_regions:
                self._region_overlay_view.set_regions(color_regions)
        
        self.statusBarMessage.emit("All regions reset to original positions")
    
    def validate_entry(self) -> bool:
        """Validate the entry of the step."""
        # Check if there is a base image
        base_image = self._app_state.base_image
        if base_image is None:
            QMessageBox.warning(
                self,
                "Arrange Regions Validation",
                "Please load an image first."
            )
            return False
        
        # Check if there are too many colors
        try:
            unique_colors = self._extract_unique_colors(base_image)
            if len(unique_colors) > 32:
                QMessageBox.warning(
                    self,
                    "Arrange Regions Validation",
                    f"Image has {len(unique_colors)} unique colors. "
                    "This step works best with 32 or fewer colors. "
                    "Consider using Color Processing to reduce the number of colors first."
                )
                return False
        except Exception as e:
            QMessageBox.critical(
                self,
                "Arrange Regions Validation",
                f"Error analyzing image: {str(e)}\n\nPlease ensure the image is properly loaded and try again."
            )
            return False
        
        return True
    
    def _on_open(self) -> None:
        """Handle open event."""
        super()._on_open()
        
        # Disable Apply To Base button
        self._btn_apply_to_base.setEnabled(False)
        
        # Initialize working image to transparent (for blank background when checkbox is unchecked)
        base_image = self._app_state.base_image
        if base_image is not None:
            h, w = base_image.shape[:2]
            transparent = np.zeros((h, w, 4), dtype=np.uint8)
            self._app_state.working_image = transparent
        
        # Build region model
        self._build_region_model()
        
        # Update palette table
        self._update_palette_table()
        
        # Connect table selection
        self.palette_table.selectionModel().selectionChanged.connect(self._on_color_selected)
        
        # Override Reset button
        try:
            self._btn_reset.clicked.disconnect()
        except:
            pass
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        
        # Connect Show Base Image checkbox to update display
        try:
            self._chk_show_base.toggled.disconnect()
        except:
            pass
        self._chk_show_base.toggled.connect(self._update_region_display)
        
        # Initialize display
        self._update_region_display()
        
        self.statusBarMessage.emit("Arrange Regions: Select a color to view and arrange its regions")
    
    def _on_close(self) -> None:
        """Handle close event."""
        super()._on_close()
        
        # Clear regions overlay
        if self._region_overlay_view:
            self._region_overlay_view.set_regions(None)
        
        # Disconnect table selection
        self.palette_table.selectionModel().selectionChanged.disconnect(self._on_color_selected)
        self._chk_show_base.toggled.disconnect(self._update_region_display)
        self._btn_reset.clicked.disconnect(self._on_reset_clicked)
        
        # Clear selection
        self._selected_color = None
    
    def _extract_unique_colors(self, image: np.ndarray) -> List[QColor]:
        """Extract unique colors from the image."""
        if image is None or len(image.shape) < 3:
            return []
        
        # Reshape image to get all pixels
        pixels = image.reshape(-1, image.shape[-1])
        
        # Get unique colors (only RGB channels, ignore alpha)
        unique_rgb = np.unique(pixels[:, :3], axis=0)
        
        # Convert to QColor list
        colors = []
        for rgb in unique_rgb:
            colors.append(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        
        return colors
