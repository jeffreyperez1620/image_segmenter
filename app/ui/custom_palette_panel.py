from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
	QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
	QPushButton, QLabel, QComboBox, QCheckBox, QSpinBox,
	QScrollArea, QFrame, QMessageBox, QColorDialog
)

from app.processing.color_simplify import create_palette_from_colors


class ColorSwatch(QLabel):
	"""A color swatch widget that can be clicked to edit or remove colors."""
	
	colorChanged = Signal(int, QColor)  # index, new_color
	colorRemoved = Signal(int)  # index
	
	def __init__(self, color: QColor, index: int, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self.color = color
		self.index = index
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
		"""Handle mouse clicks to edit or remove colors."""
		if event.button() == Qt.LeftButton:
			# Left click to edit color
			new_color = QColorDialog.getColor(self.color, self)
			if new_color.isValid() and new_color != self.color:
				self.color = new_color
				self.update_display()
				self.colorChanged.emit(self.index, new_color)
		elif event.button() == Qt.RightButton:
			# Right click to remove color
			self.colorRemoved.emit(self.index)


class CustomPalettePanel(QWidget):
	"""Panel for custom palette creation and management."""
	
	simplifyRequested = Signal()
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
		title = QLabel("Custom Palette")
		title.setStyleSheet("font-weight: bold; font-size: 14px;")
		layout.addWidget(title)
		
		# Instructions
		instructions = QLabel(
			"Click 'Add Color' to add colors to your palette, or use the eyedropper to pick colors from the image.\n"
			"Click on a color swatch to edit it, right-click to remove it."
		)
		instructions.setWordWrap(True)
		layout.addWidget(instructions)
		
		# Controls
		controls_layout = QHBoxLayout()
		
		# Add color button
		self.btn_add_color = QPushButton("Add Color")
		self.btn_add_color.clicked.connect(self._add_color)
		controls_layout.addWidget(self.btn_add_color)
		
		# Eyedropper button
		self.btn_eyedropper = QPushButton("Eyedropper")
		self.btn_eyedropper.clicked.connect(self._start_eyedropper)
		controls_layout.addWidget(self.btn_eyedropper)
		
		# Clear palette button
		self.btn_clear = QPushButton("Clear Palette")
		self.btn_clear.clicked.connect(self._clear_palette)
		controls_layout.addWidget(self.btn_clear)
		
		layout.addLayout(controls_layout)
		
		# Color swatches area
		self.swatches_layout = QGridLayout()
		self.swatches_widget = QWidget()
		self.swatches_widget.setLayout(self.swatches_layout)
		
		# Scroll area for swatches
		scroll_area = QScrollArea()
		scroll_area.setWidget(self.swatches_widget)
		scroll_area.setWidgetResizable(True)
		scroll_area.setMaximumHeight(120)
		layout.addWidget(scroll_area)
		
		# Distance metric selection
		metric_layout = QHBoxLayout()
		metric_layout.addWidget(QLabel("Distance Metric:"))
		
		self.combo_metric = QComboBox()
		self.combo_metric.addItem("LAB (Perceptual)", "lab")
		self.combo_metric.addItem("RGB (Euclidean)", "rgb")
		self.combo_metric.addItem("HSV", "hsv")
		self.combo_metric.currentTextChanged.connect(self._on_metric_changed)
		metric_layout.addWidget(self.combo_metric)
		
		layout.addLayout(metric_layout)
		
		# Preserve alpha checkbox
		self.check_preserve_alpha = QCheckBox("Preserve Alpha Channel")
		self.check_preserve_alpha.setChecked(True)
		self.check_preserve_alpha.toggled.connect(self._on_preserve_alpha_changed)
		layout.addWidget(self.check_preserve_alpha)
		
		# Statistics
		self.label_stats = QLabel("No colors in palette")
		layout.addWidget(self.label_stats)
		
		# Action buttons
		action_layout = QHBoxLayout()
		
		self.btn_simplify = QPushButton("Simplify Colors")
		self.btn_simplify.clicked.connect(self.simplifyRequested.emit)
		self.btn_simplify.setEnabled(False)
		action_layout.addWidget(self.btn_simplify)
		
		self.chk_preview = QCheckBox("Show Preview")
		self.chk_preview.toggled.connect(self.previewToggled.emit)
		action_layout.addWidget(self.chk_preview)
		
		self.btn_apply = QPushButton("Apply")
		self.btn_apply.clicked.connect(self.applyRequested.emit)
		self.btn_apply.setEnabled(False)
		action_layout.addWidget(self.btn_apply)
		
		layout.addLayout(action_layout)
		
		# Add some default colors
		self._add_default_colors()
	
	def _add_default_colors(self) -> None:
		"""Add some default colors to get started."""
		default_colors = [
			QColor(255, 0, 0),    # Red
			QColor(0, 255, 0),    # Green
			QColor(0, 0, 255),    # Blue
			QColor(255, 255, 0),  # Yellow
			QColor(255, 0, 255),  # Magenta
			QColor(0, 255, 255),  # Cyan
		]
		
		for color in default_colors:
			self._add_color_to_list(color)
	
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
	
	def _on_metric_changed(self, text: str) -> None:
		"""Handle distance metric change."""
		self._distance_metric = self.combo_metric.currentData()
	
	def _on_preserve_alpha_changed(self, checked: bool) -> None:
		"""Handle preserve alpha checkbox change."""
		self._preserve_alpha = checked
	

	
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
		self.btn_simplify.setEnabled(has_colors)
		self.btn_apply.setEnabled(has_colors)
	
	def get_palette(self) -> np.ndarray:
		"""Get the current palette as a numpy array."""
		if not self._colors:
			return np.array([[0, 0, 0]], dtype=np.uint8)
		
		colors_list = [(color.red(), color.green(), color.blue()) for color in self._colors]
		return create_palette_from_colors(colors_list)
	
	def get_distance_metric(self) -> str:
		"""Get the current distance metric."""
		return self._distance_metric
	
	def get_preserve_alpha(self) -> bool:
		"""Get the preserve alpha setting."""
		return self._preserve_alpha
	
	def add_color_from_image(self, color: QColor) -> None:
		"""Add a color picked from the image."""
		self._add_color_to_list(color)
		# Keep eyedropper button active for multiple picks
		# Don't reset the button appearance
	
	def update_statistics(self, stats: dict) -> None:
		"""Update statistics display with image information."""
		# This could show information about the original image colors
		pass
	
	def reset_eyedropper_state(self) -> None:
		"""Reset the eyedropper button to its default state."""
		self._eyedropper_active = False
		self.btn_eyedropper.setText("Eyedropper")
		self.btn_eyedropper.setStyleSheet("")
