from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
	QCheckBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QPushButton,
	QRadioButton,
	QSlider,
	QVBoxLayout,
	QWidget,
	QComboBox,
)


class BgToolsPanel(QWidget):
	modeChanged = Signal(str)
	brushSizeChanged = Signal(int)
	undoRequested = Signal()
	redoRequested = Signal()
	clearRequested = Signal()
	runGrabcutRequested = Signal()
	previewToggled = Signal(bool)
	applyCropRequested = Signal()
	aiRembgRequested = Signal()
	portraitMattingRequested = Signal()
	opacityThresholdChanged = Signal(int)

	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		layout = QVBoxLayout(self)
		layout.setAlignment(Qt.AlignTop)

		# Modes
		modes_group = QGroupBox("Modes", self)
		modes_layout = QVBoxLayout(modes_group)
		self.radio_crop = QRadioButton("Crop")
		self.radio_include = QRadioButton("Include (brush)")
		self.radio_exclude = QRadioButton("Exclude (brush)")
		self.radio_erase = QRadioButton("Erase (brush)")
		self.radio_include.setChecked(True)
		for rb in (self.radio_crop, self.radio_include, self.radio_exclude, self.radio_erase):
			modes_layout.addWidget(rb)
		layout.addWidget(modes_group)

		# Brush size
		brush_group = QGroupBox("Brush", self)
		brush_form = QFormLayout(brush_group)
		self.slider_brush = QSlider(Qt.Horizontal)
		self.slider_brush.setMinimum(3)
		self.slider_brush.setMaximum(100)
		self.slider_brush.setValue(24)
		brush_form.addRow("Size", self.slider_brush)
		layout.addWidget(brush_group)

		# Opacity threshold for AI background removal
		opacity_group = QGroupBox("AI Opacity Threshold", self)
		opacity_form = QFormLayout(opacity_group)
		self.slider_opacity_threshold = QSlider(Qt.Horizontal)
		self.slider_opacity_threshold.setMinimum(0)
		self.slider_opacity_threshold.setMaximum(255)
		self.slider_opacity_threshold.setValue(128)
		opacity_form.addRow("Threshold", self.slider_opacity_threshold)
		layout.addWidget(opacity_group)

		# Actions
		actions_row = QHBoxLayout()
		self.btn_undo = QPushButton("Undo")
		self.btn_redo = QPushButton("Redo")
		self.btn_clear = QPushButton("Clear")
		actions_row.addWidget(self.btn_undo)
		actions_row.addWidget(self.btn_redo)
		actions_row.addWidget(self.btn_clear)
		layout.addLayout(actions_row)

		# AI remove background with model chooser
		ai_row = QHBoxLayout()
		self.combo_ai_model = QComboBox()
		self._populate_ai_models()
		self.btn_ai = QPushButton("AI Remove Background (rembg)")
		ai_row.addWidget(self.combo_ai_model, 1)
		ai_row.addWidget(self.btn_ai, 2)
		layout.addLayout(ai_row)

		# Portrait matting refine
		self.btn_portrait = QPushButton("Portrait Matting (refine)")
		layout.addWidget(self.btn_portrait)

		# Run + preview
		run_row = QHBoxLayout()
		self.btn_run = QPushButton("Run Background Removal")
		self.chk_preview = QCheckBox("Show Preview")
		run_row.addWidget(self.btn_run)
		run_row.addWidget(self.chk_preview)
		layout.addLayout(run_row)

		# Crop apply
		self.btn_apply_crop = QPushButton("Apply Crop")
		layout.addWidget(self.btn_apply_crop)

		layout.addStretch(1)

		# Wire signals
		self.radio_crop.toggled.connect(self._emit_mode)
		self.radio_include.toggled.connect(self._emit_mode)
		self.radio_exclude.toggled.connect(self._emit_mode)
		self.radio_erase.toggled.connect(self._emit_mode)
		self.slider_brush.valueChanged.connect(self.brushSizeChanged)
		self.slider_opacity_threshold.valueChanged.connect(self.opacityThresholdChanged)
		self.btn_undo.clicked.connect(self.undoRequested)
		self.btn_redo.clicked.connect(self.redoRequested)
		self.btn_clear.clicked.connect(self.clearRequested)
		self.btn_run.clicked.connect(self.runGrabcutRequested)
		self.btn_ai.clicked.connect(self.aiRembgRequested)
		self.btn_portrait.clicked.connect(self.portraitMattingRequested)
		self.chk_preview.toggled.connect(self.previewToggled)
		self.btn_apply_crop.clicked.connect(self.applyCropRequested)

		# Initialize mode
		self._emit_mode()

	def _emit_mode(self) -> None:
		if self.radio_crop.isChecked():
			self.modeChanged.emit("crop")
			return
		if self.radio_include.isChecked():
			self.modeChanged.emit("include")
			return
		if self.radio_exclude.isChecked():
			self.modeChanged.emit("exclude")
			return
		if self.radio_erase.isChecked():
			self.modeChanged.emit("erase")
			return

	def _populate_ai_models(self) -> None:
		# text shown -> model id in userData
		options = [
			("isnet-general-use (general)", "isnet-general-use"),
			("u2net (general, detailed)", "u2net"),
			("u2netp (fast, small)", "u2netp"),
			("u2net_human_seg (people)", "u2net_human_seg"),
			("u2net_cloth_seg (clothing)", "u2net_cloth_seg"),
			("isnet-anime (anime)", "isnet-anime"),
			("silueta (simple)", "silueta"),
		]
		for label, mid in options:
			self.combo_ai_model.addItem(label, mid)
		self.combo_ai_model.setCurrentIndex(0)

	def get_selected_rembg_model(self) -> str:
		data = self.combo_ai_model.currentData()
		return data if isinstance(data, str) else "isnet-general-use"

	def get_opacity_threshold(self) -> int:
		return self.slider_opacity_threshold.value()
