from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
	QCheckBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QPushButton,
	QSlider,
	QVBoxLayout,
	QWidget,
	QComboBox,
	QLabel,
	QSpinBox,
)


class ColorSimplifyPanel(QWidget):
	simplifyRequested = Signal()
	previewToggled = Signal(bool)
	applyRequested = Signal()
	
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		layout = QVBoxLayout(self)
		layout.setAlignment(Qt.AlignTop)
		
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
		color_form.addRow("Algorithm", self.combo_algorithm)
		
		# Alpha preservation
		self.chk_preserve_alpha = QCheckBox("Preserve Alpha Channel")
		self.chk_preserve_alpha.setChecked(True)
		color_form.addRow("", self.chk_preserve_alpha)
		
		layout.addWidget(color_group)
		
		# Statistics display
		stats_group = QGroupBox("Color Statistics", self)
		stats_layout = QVBoxLayout(stats_group)
		self.label_stats = QLabel("No image loaded")
		self.label_stats.setWordWrap(True)
		stats_layout.addWidget(self.label_stats)
		layout.addWidget(stats_group)
		
		# Actions
		actions_row = QHBoxLayout()
		self.btn_simplify = QPushButton("Simplify Colors")
		self.chk_preview = QCheckBox("Show Preview")
		actions_row.addWidget(self.btn_simplify)
		actions_row.addWidget(self.chk_preview)
		layout.addLayout(actions_row)
		
		# Apply button
		self.btn_apply = QPushButton("Apply Simplification")
		layout.addWidget(self.btn_apply)
		
		layout.addStretch(1)
		
		# Wire signals
		self.btn_simplify.clicked.connect(self.simplifyRequested)
		self.chk_preview.toggled.connect(self.previewToggled)
		self.btn_apply.clicked.connect(self.applyRequested)
	
	def get_num_colors(self) -> int:
		"""Get the target number of colors."""
		return self.spin_num_colors.value()
	
	def get_algorithm(self) -> str:
		"""Get the selected algorithm."""
		return self.combo_algorithm.currentData()
	
	def get_preserve_alpha(self) -> bool:
		"""Get whether to preserve alpha channel."""
		return self.chk_preserve_alpha.isChecked()
	
	def update_statistics(self, stats: dict) -> None:
		"""Update the statistics display."""
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
