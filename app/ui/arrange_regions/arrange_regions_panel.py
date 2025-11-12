"""
Panel for arranging regions by color.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSizePolicy, QCheckBox, QComboBox, QLineEdit, QFormLayout, QFileDialog
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
        # Track if any regions have been moved or rotated from their original positions
        self._has_region_changes = False
        super().__init__(parent, app_state, image_view)
        
        # Region model: Dict[color_tuple, ColorRegions]
        self._region_model: Dict[Tuple[int, int, int], ColorRegions] = {}
        
        # Currently selected color
        self._selected_color: Optional[Tuple[int, int, int]] = None
        
        # Region overlay view
        self._region_overlay_view: Optional[RegionOverlayView] = None
        
        # Export settings
        self._lock_aspect_ratio = True
        self._export_units = "in"
        self._export_width_pixels = 0
        self._export_height_pixels = 0
        self._aspect_ratio = 1.0
        self._export_margin_width = 0.0
        self._export_margin_height = 0.0
        self._export_enable_smoothing = False
        
        self._init_ui()
        
        # Create region overlay view after UI is initialized
        if self._image_view:
            self._region_overlay_view = RegionOverlayView(
                parent=self._image_view,
                app_state=self._app_state,
                image_view=self._image_view,
                region_model=self._region_model,
                panel=self
            )
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Color palette table
        palette_group = QGroupBox("Color Palette")
        palette_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        palette_layout = QVBoxLayout(palette_group)
        
        self.palette_scroll = QScrollArea()
        self.palette_scroll.setWidgetResizable(True)
        self.palette_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.palette_table = QTableWidget()
        self.palette_table.setColumnCount(2)
        self.palette_table.setHorizontalHeaderLabels(["Color", "Info"])
        self.palette_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.palette_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.palette_table.setColumnWidth(0, 50)
        self.palette_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.palette_table.setSelectionMode(QTableWidget.SingleSelection)
        
        self.palette_scroll.setWidget(self.palette_table)
        palette_layout.addWidget(self.palette_scroll)
        layout.addWidget(palette_group)
        
        # Export to SVG section
        export_group = QGroupBox("Export to SVG")
        export_layout = QVBoxLayout(export_group)
        
        # First row: Lock Aspect Ratio checkbox and Units combo box
        aspect_units_layout = QHBoxLayout()
        aspect_units_layout.addWidget(QLabel("Lock Aspect Ratio:"))
        self._chk_lock_aspect = QCheckBox()
        self._chk_lock_aspect.setChecked(True)
        aspect_units_layout.addWidget(self._chk_lock_aspect)
        aspect_units_layout.addStretch()
        aspect_units_layout.addWidget(QLabel("Units:"))
        self._combo_units = QComboBox()
        self._combo_units.addItems(["in", "cm", "mm"])
        aspect_units_layout.addWidget(self._combo_units)
        export_layout.addLayout(aspect_units_layout)
        
        # Second row: Width and Height inputs
        dimensions_layout = QHBoxLayout()
        dimensions_layout.addWidget(QLabel("Width:"))
        self._edit_width = QLineEdit()
        self._edit_width.setPlaceholderText("Width")
        dimensions_layout.addWidget(self._edit_width)
        dimensions_layout.addWidget(QLabel("Height:"))
        self._edit_height = QLineEdit()
        self._edit_height.setPlaceholderText("Height")
        dimensions_layout.addWidget(self._edit_height)
        export_layout.addLayout(dimensions_layout)
        
        # Third row: Margin Width and Margin Height inputs
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Margin Width:"))
        self._edit_margin_width = QLineEdit()
        self._edit_margin_width.setPlaceholderText("Margin Width")
        margin_layout.addWidget(self._edit_margin_width)
        margin_layout.addWidget(QLabel("Margin Height:"))
        self._edit_margin_height = QLineEdit()
        self._edit_margin_height.setPlaceholderText("Margin Height")
        margin_layout.addWidget(self._edit_margin_height)
        export_layout.addLayout(margin_layout)
        
        # Fourth row: Enable Smoothing checkbox
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Enable Smoothing:"))
        self._chk_enable_smoothing = QCheckBox()
        self._chk_enable_smoothing.setChecked(False)
        smoothing_layout.addWidget(self._chk_enable_smoothing)
        smoothing_layout.addStretch()
        export_layout.addLayout(smoothing_layout)
        
        # Export buttons at the bottom
        export_buttons_layout = QHBoxLayout()
        self._btn_export_selected = QPushButton("Export Selected")
        self._btn_export_all = QPushButton("Export All")
        export_buttons_layout.addWidget(self._btn_export_selected)
        export_buttons_layout.addWidget(self._btn_export_all)
        export_layout.addLayout(export_buttons_layout)
        
        layout.addWidget(export_group)
        
        # Add stretch at the end
        layout.addStretch()
        
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
        # Clear active region when selection changes
        if self._region_overlay_view is not None and self._region_overlay_view._active_item:
            self._region_overlay_view._active_item.set_active(False)
            self._region_overlay_view._active_item = None
        
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
        if self._region_overlay_view is None:
            return
        
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
        
        # Restore original positions and rotations for all regions
        for color_regions in self._region_model.values():
            for region in color_regions.regions:
                region.position = region.original_position
                region.rotation = region.original_rotation
        
        # Mark changes as applied (regions reset to original)
        self._has_region_changes = False
        self.mark_changes_applied()
        
        # Update display if a color is selected
        if self._selected_color and self._region_overlay_view:
            # Refresh the overlay to show updated positions
            color_regions = self._region_model.get(self._selected_color)
            if color_regions:
                self._region_overlay_view.set_regions(color_regions)
        
        self.statusBarMessage.emit("All regions reset to original positions")
    
    def _on_region_changed(self) -> None:
        """Called when a region is moved or rotated."""
        self._has_region_changes = True
        self.mark_changes_unapplied()
    
    def has_unapplied_changes(self) -> bool:
        """Override to check if any regions have been moved or rotated."""
        return self._has_region_changes
    
    def mark_changes_applied(self) -> None:
        """Override to also clear region changes flag."""
        super().mark_changes_applied()
        self._has_region_changes = False
    
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
        
        # Reset region changes tracking
        self._has_region_changes = False
        self.mark_changes_applied()
        
        # Disable Apply To Base button
        self._btn_apply_to_base.setEnabled(False)
        
        # Initialize working image to transparent (for blank background when checkbox is unchecked)
        base_image = self._app_state.base_image
        if base_image is not None:
            h, w = base_image.shape[:2]
            transparent = np.zeros((h, w, 4), dtype=np.uint8)
            self._app_state.working_image = transparent
        
        # Activate overlay
        if self._region_overlay_view is not None:
            self._region_overlay_view.set_active(True)
        
        # Build region model
        self._build_region_model()
        
        # Update overlay view with region model reference
        if self._region_overlay_view is not None:
            self._region_overlay_view._region_model = self._region_model
        
        # Update palette table
        self._update_palette_table()
        
        # Connect table selection
        self.palette_table.selectionModel().selectionChanged.connect(self._on_color_selected)
        
        # Override Reset button - disconnect all first, then connect
        self._btn_reset.clicked.disconnect()
        self._btn_reset.clicked.connect(self._on_reset_clicked)
        
        # Connect Show Base Image checkbox to update display - disconnect all first
        self._chk_show_base.toggled.disconnect()
        self._chk_show_base.toggled.connect(self._update_region_display)
        
        # Calculate bounding box of non-transparent pixels
        self._calculate_export_dimensions()
        
        # Connect export UI signals
        self._chk_lock_aspect.toggled.connect(self._on_lock_aspect_toggled)
        self._edit_width.textChanged.connect(self._on_width_changed)
        self._edit_height.textChanged.connect(self._on_height_changed)
        self._edit_margin_width.textChanged.connect(self._on_margin_width_changed)
        self._edit_margin_height.textChanged.connect(self._on_margin_height_changed)
        self._combo_units.currentTextChanged.connect(self._on_units_changed)
        self._btn_export_selected.clicked.connect(self._on_export_selected)
        self._btn_export_all.clicked.connect(self._on_export_all)
        
        # Initialize display
        self._update_region_display()
        
        self.statusBarMessage.emit("Arrange Regions: Select a color to view and arrange its regions")
    
    def _update_controls_state(self):
        """Override to keep Apply to Base disabled (regions are not saved to base)."""
        has_base = self._app_state.base_image is not None
        
        # Always disable Apply to Base - regions are just for arrangement, not for saving
        self._btn_apply_to_base.setEnabled(False)
    
    def _on_close(self) -> None:
        """Handle close event."""
        super()._on_close()
        
        # Deactivate overlay and clear regions
        if self._region_overlay_view:
            self._region_overlay_view.set_active(False)
            self._region_overlay_view.set_regions(None)
        
        # Disconnect table selection and other signals
        self.palette_table.selectionModel().selectionChanged.disconnect(self._on_color_selected)
        
        # Clear selection
        self.palette_table.clearSelection()
        self._selected_color = None
    
    def _extract_unique_colors(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract unique colors from the image."""
        if len(image.shape) == 2:
            # Grayscale
            unique_vals = np.unique(image)
            return [(int(v), int(v), int(v)) for v in unique_vals]
        elif len(image.shape) == 3:
            # Color image
            if image.shape[2] == 4:
                # RGBA - ignore alpha for color comparison
                rgb = image[:, :, :3]
            else:
                rgb = image
            # Reshape to list of pixels
            pixels = rgb.reshape(-1, rgb.shape[-1])
            # Get unique colors
            unique_colors = np.unique(pixels, axis=0)
            return [tuple(int(c) for c in color) for color in unique_colors]
        return []
    
    def _calculate_export_dimensions(self) -> None:
        """Calculate the bounding box of non-transparent pixels and set export dimensions."""
        base_image = self._app_state.base_image
        if base_image is None:
            self._export_width_pixels = 0
            self._export_height_pixels = 0
            self._aspect_ratio = 1.0
            return
        
        # Get alpha channel
        if base_image.shape[2] >= 4:
            alpha = base_image[:, :, 3]
        else:
            # No alpha channel, use full image
            self._export_width_pixels = base_image.shape[1]
            self._export_height_pixels = base_image.shape[0]
            self._aspect_ratio = self._export_width_pixels / self._export_height_pixels if self._export_height_pixels > 0 else 1.0
            self._update_export_fields()
            return
        
        # Find non-transparent pixels
        non_transparent = alpha > 0
        
        if not np.any(non_transparent):
            # All transparent, use full image dimensions
            self._export_width_pixels = base_image.shape[1]
            self._export_height_pixels = base_image.shape[0]
            self._aspect_ratio = 1.0
            self._update_export_fields()
            return
        
        # Find bounding box of non-transparent pixels
        rows = np.any(non_transparent, axis=1)
        cols = np.any(non_transparent, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # No non-transparent pixels found
            self._export_width_pixels = base_image.shape[1]
            self._export_height_pixels = base_image.shape[0]
            self._aspect_ratio = 1.0
            self._update_export_fields()
            return
        
        y_min = np.argmax(rows)
        y_max = len(rows) - np.argmax(rows[::-1])
        x_min = np.argmax(cols)
        x_max = len(cols) - np.argmax(cols[::-1])
        
        # Calculate dimensions
        self._export_width_pixels = x_max - x_min
        self._export_height_pixels = y_max - y_min
        self._aspect_ratio = self._export_width_pixels / self._export_height_pixels if self._export_height_pixels > 0 else 1.0
        
        self._update_export_fields()
    
    def _update_export_fields(self) -> None:
        """Update the width and height fields with current pixel dimensions converted to selected units."""
        if self._export_width_pixels == 0 or self._export_height_pixels == 0:
            self._edit_width.clear()
            self._edit_height.clear()
            return
        
        # Convert pixels to selected units (assuming 96 DPI for now)
        dpi = 96.0
        pixels_per_inch = dpi
        
        # Convert to inches first
        width_inches = self._export_width_pixels / pixels_per_inch
        height_inches = self._export_height_pixels / pixels_per_inch
        
        # Convert to selected unit
        unit = self._combo_units.currentText()
        if unit == "cm":
            width_value = width_inches * 2.54
            height_value = height_inches * 2.54
        elif unit == "mm":
            width_value = width_inches * 25.4
            height_value = height_inches * 25.4
        else:  # "in"
            width_value = width_inches
            height_value = height_inches
        
        # Update fields (block signals to avoid recursion)
        self._edit_width.blockSignals(True)
        self._edit_height.blockSignals(True)
        self._edit_width.setText(f"{width_value:.2f}")
        self._edit_height.setText(f"{height_value:.2f}")
        self._edit_width.blockSignals(False)
        self._edit_height.blockSignals(False)
    
    def _on_lock_aspect_toggled(self, checked: bool) -> None:
        """Handle lock aspect ratio checkbox toggle."""
        self._lock_aspect_ratio = checked
    
    def _on_units_changed(self, unit: str) -> None:
        """Handle units combo box change."""
        self._export_units = unit
        self._update_export_fields()
    
    def _on_width_changed(self, text: str) -> None:
        """Handle width input change."""
        if not text:
            return
        
        try:
            width_value = float(text)
            if width_value <= 0:
                return
            
            # Convert to pixels
            dpi = 96.0
            pixels_per_inch = dpi
            unit = self._combo_units.currentText()
            
            if unit == "cm":
                width_inches = width_value / 2.54
            elif unit == "mm":
                width_inches = width_value / 25.4
            else:  # "in"
                width_inches = width_value
            
            new_width_pixels = width_inches * pixels_per_inch
            
            if self._lock_aspect_ratio:
                # Calculate new height based on aspect ratio
                new_height_pixels = new_width_pixels / self._aspect_ratio if self._aspect_ratio > 0 else 0
                self._export_height_pixels = new_height_pixels
                
                # Update height field
                height_inches = new_height_pixels / pixels_per_inch
                if unit == "cm":
                    height_value = height_inches * 2.54
                elif unit == "mm":
                    height_value = height_inches * 25.4
                else:  # "in"
                    height_value = height_inches
                
                self._edit_height.blockSignals(True)
                self._edit_height.setText(f"{height_value:.2f}")
                self._edit_height.blockSignals(False)
            else:
                self._export_width_pixels = new_width_pixels
        except ValueError:
            pass
    
    def _on_height_changed(self, text: str) -> None:
        """Handle height input change."""
        if not text:
            return
        
        try:
            height_value = float(text)
            if height_value <= 0:
                return
            
            # Convert to pixels
            dpi = 96.0
            pixels_per_inch = dpi
            unit = self._combo_units.currentText()
            
            if unit == "cm":
                height_inches = height_value / 2.54
            elif unit == "mm":
                height_inches = height_value / 25.4
            else:  # "in"
                height_inches = height_value
            
            new_height_pixels = height_inches * pixels_per_inch
            
            if self._lock_aspect_ratio:
                # Calculate new width based on aspect ratio
                new_width_pixels = new_height_pixels * self._aspect_ratio
                self._export_width_pixels = new_width_pixels
                
                # Update width field
                width_inches = new_width_pixels / pixels_per_inch
                if unit == "cm":
                    width_value = width_inches * 2.54
                elif unit == "mm":
                    width_value = width_inches * 25.4
                else:  # "in"
                    width_value = width_inches
                
                self._edit_width.blockSignals(True)
                self._edit_width.setText(f"{width_value:.2f}")
                self._edit_width.blockSignals(False)
            else:
                self._export_height_pixels = new_height_pixels
        except ValueError:
            pass
    
    def _on_margin_width_changed(self, text: str) -> None:
        """Handle margin width input change."""
        if not text:
            self._export_margin_width = 0.0
            return
        
        try:
            margin_value = float(text)
            if margin_value < 0:
                return
            self._export_margin_width = margin_value
        except ValueError:
            pass
    
    def _on_margin_height_changed(self, text: str) -> None:
        """Handle margin height input change."""
        if not text:
            self._export_margin_height = 0.0
            return
        
        try:
            margin_value = float(text)
            if margin_value < 0:
                return
            self._export_margin_height = margin_value
        except ValueError:
            pass
    
    def _on_export_selected(self) -> None:
        """Handle Export Selected button click."""
        if self._selected_color is None:
            QMessageBox.warning(self, "Export", "Please select a color to export.")
            return
        
        # Get the selected color's regions
        color_regions = self._region_model.get(self._selected_color)
        if not color_regions or not color_regions.regions:
            QMessageBox.warning(self, "Export", "No regions found for selected color.")
            return
        
        # Get export parameters
        try:
            total_width_value = float(self._edit_width.text())
            total_height_value = float(self._edit_height.text())
            margin_width_value = float(self._edit_margin_width.text()) if self._edit_margin_width.text() else 0.0
            margin_height_value = float(self._edit_margin_height.text()) if self._edit_margin_height.text() else 0.0
            units = self._combo_units.currentText()
            enable_smoothing = self._chk_enable_smoothing.isChecked()
        except ValueError:
            QMessageBox.warning(self, "Export", "Please enter valid numeric values for dimensions and margins.")
            return
        
        # Calculate the bounding box of the regions being exported
        from processing.svg_export import calculate_regions_bounding_box
        regions_bbox = calculate_regions_bounding_box(color_regions.regions)
        if regions_bbox is None:
            QMessageBox.warning(self, "Export", "Failed to calculate regions bounding box.")
            return
        
        regions_min_x, regions_min_y, regions_width, regions_height = regions_bbox
        
        # Get total image dimensions (non-transparent bounds)
        total_image_width_pixels = self._export_width_pixels
        total_image_height_pixels = self._export_height_pixels
        
        if total_image_width_pixels <= 0 or total_image_height_pixels <= 0:
            QMessageBox.warning(self, "Export", "Invalid total image dimensions.")
            return
        
        # Calculate scale factor based on total image size
        # Convert total dimensions to pixels (assuming 96 DPI)
        dpi = 96.0
        pixels_per_inch = dpi
        if units == "cm":
            total_width_pixels = (total_width_value / 2.54) * pixels_per_inch
            total_height_pixels = (total_height_value / 2.54) * pixels_per_inch
        elif units == "mm":
            total_width_pixels = (total_width_value / 25.4) * pixels_per_inch
            total_height_pixels = (total_height_value / 25.4) * pixels_per_inch
        else:  # "in"
            total_width_pixels = total_width_value * pixels_per_inch
            total_height_pixels = total_height_value * pixels_per_inch
        
        # Calculate scale factors
        scale_x = total_width_pixels / total_image_width_pixels
        scale_y = total_height_pixels / total_image_height_pixels
        scale = min(scale_x, scale_y)  # Maintain aspect ratio
        
        # Calculate actual output dimensions for the exported regions
        output_width = regions_width * scale
        output_height = regions_height * scale
        
        # Convert back to the specified units
        if units == "cm":
            output_width = (output_width / pixels_per_inch) * 2.54
            output_height = (output_height / pixels_per_inch) * 2.54
        elif units == "mm":
            output_width = (output_width / pixels_per_inch) * 25.4
            output_height = (output_height / pixels_per_inch) * 25.4
        else:  # "in"
            output_width = output_width / pixels_per_inch
            output_height = output_height / pixels_per_inch
        
        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Regions to SVG",
            "regions.svg",
            "SVG Files (*.svg);;All Files (*)"
        )
        
        if not file_path:
            # User cancelled
            return
        
        # Export to SVG
        from processing.svg_export import export_regions_to_svg
        success = export_regions_to_svg(
            color_regions.regions,
            output_width,
            output_height,
            margin_width_value,
            margin_height_value,
            units,
            file_path,
            enable_smoothing
        )
        
        if success:
            self.statusBarMessage.emit(f"Regions exported to: {file_path}")
        else:
            QMessageBox.warning(self, "Export", "Failed to export regions to SVG.")
    
    def _on_export_all(self) -> None:
        """Handle Export All button click."""
        if not self._region_model:
            QMessageBox.warning(self, "Export", "No regions available to export.")
            return
        
        # Get export parameters
        try:
            total_width_value = float(self._edit_width.text())
            total_height_value = float(self._edit_height.text())
            margin_width_value = float(self._edit_margin_width.text()) if self._edit_margin_width.text() else 0.0
            margin_height_value = float(self._edit_margin_height.text()) if self._edit_margin_height.text() else 0.0
            units = self._combo_units.currentText()
            enable_smoothing = self._chk_enable_smoothing.isChecked()
        except ValueError:
            QMessageBox.warning(self, "Export", "Please enter valid numeric values for dimensions and margins.")
            return
        
        # Collect all regions from all colors
        all_regions = []
        for color_regions in self._region_model.values():
            all_regions.extend(color_regions.regions)
        
        if not all_regions:
            QMessageBox.warning(self, "Export", "No regions found to export.")
            return
        
        # Calculate the bounding box of the regions being exported
        from processing.svg_export import calculate_regions_bounding_box
        regions_bbox = calculate_regions_bounding_box(all_regions)
        if regions_bbox is None:
            QMessageBox.warning(self, "Export", "Failed to calculate regions bounding box.")
            return
        
        regions_min_x, regions_min_y, regions_width, regions_height = regions_bbox
        
        # Get total image dimensions (non-transparent bounds)
        total_image_width_pixels = self._export_width_pixels
        total_image_height_pixels = self._export_height_pixels
        
        if total_image_width_pixels <= 0 or total_image_height_pixels <= 0:
            QMessageBox.warning(self, "Export", "Invalid total image dimensions.")
            return
        
        # Calculate scale factor based on total image size
        # Convert total dimensions to pixels (assuming 96 DPI)
        dpi = 96.0
        pixels_per_inch = dpi
        if units == "cm":
            total_width_pixels = (total_width_value / 2.54) * pixels_per_inch
            total_height_pixels = (total_height_value / 2.54) * pixels_per_inch
        elif units == "mm":
            total_width_pixels = (total_width_value / 25.4) * pixels_per_inch
            total_height_pixels = (total_height_value / 25.4) * pixels_per_inch
        else:  # "in"
            total_width_pixels = total_width_value * pixels_per_inch
            total_height_pixels = total_height_value * pixels_per_inch
        
        # Calculate scale factors
        scale_x = total_width_pixels / total_image_width_pixels
        scale_y = total_height_pixels / total_image_height_pixels
        scale = min(scale_x, scale_y)  # Maintain aspect ratio
        
        # Calculate actual output dimensions for the exported regions
        output_width = regions_width * scale
        output_height = regions_height * scale
        
        # Convert back to the specified units
        if units == "cm":
            output_width = (output_width / pixels_per_inch) * 2.54
            output_height = (output_height / pixels_per_inch) * 2.54
        elif units == "mm":
            output_width = (output_width / pixels_per_inch) * 25.4
            output_height = (output_height / pixels_per_inch) * 25.4
        else:  # "in"
            output_width = output_width / pixels_per_inch
            output_height = output_height / pixels_per_inch
        
        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export All Regions to SVG",
            "all_regions.svg",
            "SVG Files (*.svg);;All Files (*)"
        )
        
        if not file_path:
            # User cancelled
            return
        
        # Export to SVG
        from processing.svg_export import export_regions_to_svg
        success = export_regions_to_svg(
            all_regions,
            output_width,
            output_height,
            margin_width_value,
            margin_height_value,
            units,
            file_path,
            enable_smoothing
        )
        
        if success:
            self.statusBarMessage.emit(f"All regions exported to: {file_path}")
        else:
            QMessageBox.warning(self, "Export", "Failed to export regions to SVG.")
