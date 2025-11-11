from __future__ import annotations

from typing import Optional
import numpy as np

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
	QLabel,
)

from model import AppState
from ui.background_removal.background_removal_view import BackgroundRemovalView
from ui.base_step import BaseStep
from ui.image_view import ImageView
from utils.qt_image import numpy_rgba_to_qimage, qimage_to_numpy_bgr, composite_foreground_over_transparent
from processing.grabcut import apply_grabcut
from processing.rembg_infer import rembg_remove_bgr_to_rgba
from utils.transparency_utils import clamp_alpha_channel
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt


class BgToolsPanel(BaseStep):

	def __init__(self, parent: Optional[QWidget] = None, app_state: Optional[AppState] = None, image_view: Optional[ImageView] = None) -> None:
		super().__init__(parent, app_state, image_view)
		
		# Create the main content widget
		main_widget = QWidget()
		layout = QVBoxLayout(main_widget)
		layout.setAlignment(Qt.AlignTop)

		# Background removal view for handling masks and crop rectangles
		self._background_removal_view = BackgroundRemovalView(self._image_view, self._app_state, self._image_view)

		# Automatic (AI) Background Removal Group
		automatic_group = QGroupBox("Automatic (AI) Background Removal", main_widget)
		automatic_layout = QVBoxLayout(automatic_group)
		
		# AI model selection and execution
		ai_row = QHBoxLayout()
		self.combo_ai_model = QComboBox()
		self._populate_ai_models()
		self.btn_ai = QPushButton("AI Remove Background")
		ai_row.addWidget(QLabel("Model:"))
		ai_row.addWidget(self.combo_ai_model, 1)
		ai_row.addWidget(self.btn_ai, 2)
		automatic_layout.addLayout(ai_row)
		
		# AI Opacity Threshold
		opacity_form = QFormLayout()
		self.slider_opacity_threshold = QSlider(Qt.Horizontal)
		self.slider_opacity_threshold.setMinimum(0)
		self.slider_opacity_threshold.setMaximum(255)
		self.slider_opacity_threshold.setValue(128)
		opacity_form.addRow("AI Opacity Threshold", self.slider_opacity_threshold)
		automatic_layout.addLayout(opacity_form)
		
		layout.addWidget(automatic_group)

		# Manual Background Removal Group
		manual_group = QGroupBox("Manual Background Removal", main_widget)
		manual_layout = QVBoxLayout(manual_group)
		
		# Modes with Apply Crop button next to Crop radio
		modes_row = QHBoxLayout()
		modes_layout = QVBoxLayout()
		self.radio_crop = QRadioButton("Crop")
		self.radio_include = QRadioButton("Include (brush)")
		self.radio_exclude = QRadioButton("Exclude (brush)")
		self.radio_erase = QRadioButton("Erase (brush)")
		self.radio_include.setChecked(True)
		for rb in (self.radio_crop, self.radio_include, self.radio_exclude, self.radio_erase):
			modes_layout.addWidget(rb)
		modes_row.addLayout(modes_layout)
		
		# Apply Crop button next to Crop radio
		self.btn_apply_crop = QPushButton("Apply Crop")
		modes_row.addWidget(self.btn_apply_crop)
		manual_layout.addLayout(modes_row)

		# Brush size
		brush_form = QFormLayout()
		self.slider_brush = QSlider(Qt.Horizontal)
		self.slider_brush.setMinimum(3)
		self.slider_brush.setMaximum(100)
		self.slider_brush.setValue(24)
		brush_form.addRow("Brush Size", self.slider_brush)
		manual_layout.addLayout(brush_form)

		# Manual actions
		actions_row = QHBoxLayout()
		self.btn_undo = QPushButton("Undo")
		self.btn_redo = QPushButton("Redo")
		self.btn_clear = QPushButton("Clear")
		actions_row.addWidget(self.btn_undo)
		actions_row.addWidget(self.btn_redo)
		actions_row.addWidget(self.btn_clear)
		manual_layout.addLayout(actions_row)

		# Manual background removal button
		self.btn_run = QPushButton("Run Background Removal")
		manual_layout.addWidget(self.btn_run)
		
		layout.addWidget(manual_group)
		layout.addStretch(1)

		self._working_ai_image = None

		# Set the main widget
		self.set_main_widget(main_widget)

		# Initialize mode
		self._handle_mode_change()
	
	def _handle_mode_change(self) -> None:
		if self.radio_crop.isChecked():
			mode = "crop"
		elif self.radio_include.isChecked():
			mode = "include"
		elif self.radio_exclude.isChecked():
			mode = "exclude"
		elif self.radio_erase.isChecked():
			mode = "erase"
		else:
			mode = "none"
		
		"""Handle background removal mode changes."""
		# Update the background removal view
		self._background_removal_view.set_mode(mode)
		
		self._image_view.set_mode("none")  # Allow normal interaction
		# Update the image view cursor based on mode
		if mode == "none":
			self._image_view._view.viewport().unsetCursor()
		else:
			self._image_view._view.viewport().setCursor(Qt.CrossCursor)

	def _on_opacity_threshold_changed(self, threshold: int) -> None:
		"""Handle opacity threshold changes by re-processing the AI output."""
		# Use the original AI image if available, otherwise use the working image
		if self._working_ai_image is not None:
			# Apply thresholding logic here instead of in ImageView
			rgba_working = np.array(self._working_ai_image, copy=True)
			
			# Apply opacity threshold: pixels above threshold become fully opaque (255),
			# pixels at or below threshold become fully transparent (0)
			alpha = rgba_working[:, :, 3]
			above_threshold = alpha > threshold
			rgba_working[above_threshold, 3] = 255
			rgba_working[~above_threshold, 3] = 0
			
			# Convert to QImage and set as temporary display
			from utils.qt_image import ensure_qimage
			qimg = ensure_qimage(rgba_working)
			self.set_working_image(rgba_working)

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

	def _on_run_grabcut(self) -> None:
		# Get working image from app state
		working_image = self._app_state.working_image
		if working_image is None:
			return
		# Show busy indicator
		# Note: statusBar() would need to be passed from MainWindow
		self.btn_run.setEnabled(False)
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			# Convert RGBA to BGR for grabcut processing
			bgr_full = working_image[:, :, [2, 1, 0]]  # Convert RGBA to BGR
			h_full, w_full = bgr_full.shape[:2]
			rect = self._background_removal_view.get_crop_rect_xywh()
			user_mask_full = self._background_removal_view.get_user_mask()

			def _bbox_of(mask: np.ndarray, value: int) -> Optional[tuple[int, int, int, int]]:
				rows, cols = np.where(mask == value)
				if rows.size == 0:
					return None
				ymin, ymax = int(rows.min()), int(rows.max())
				xmin, xmax = int(cols.min()), int(cols.max())
				return (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)

			if rect is not None:
				rx, ry, rw, rh = rect
				rx = max(0, min(rx, w_full - 1))
				ry = max(0, min(ry, h_full - 1))
				rw = max(1, min(rw, w_full - rx))
				rh = max(1, min(rh, h_full - ry))
				bgr = bgr_full[ry:ry+rh, rx:rx+rw]
				init_mask_roi: Optional[np.ndarray] = None
				if user_mask_full is not None:
					um = user_mask_full[ry:ry+rh, rx:rx+rw]
					if (um == 1).any() or (um == 2).any():
						init_mask_roi = np.zeros((rh, rw), dtype=np.uint8)
						init_mask_roi[um == 1] = 3  # PR_FGD
						init_mask_roi[um == 2] = 2  # PR_BGD
						
						# For better GrabCut performance, set a reasonable initial guess
						# If we have include marks, create a bounding box around them and set as PR_FGD
						bbox = _bbox_of(um, 1)
						if bbox is not None:
							sx, sy, sw, sh = bbox
							# Expand the bounding box slightly for better initialization
							margin = max(5, min(rw, rh) // 20)
							sx = max(0, sx - margin)
							sy = max(0, sy - margin)
							sw = min(rw - sx, sw + 2 * margin)
							sh = min(rh - sy, sh + 2 * margin)
							# Set the expanded area as PR_FGD (probably foreground)
							init_mask_roi[sy:sy+sh, sx:sx+sw] = 3
				# Validate seeds
				use_mask_init = False
				if init_mask_roi is not None:
					has_prfgd = (init_mask_roi == 3).any()
					has_prbgd = (init_mask_roi == 2).any()
					# Use mask initialization if we have any foreground marks, even without background marks
					use_mask_init = has_prfgd
				if use_mask_init:
					gc_mask_roi = apply_grabcut(bgr, rect_xywh=None, init_mask=init_mask_roi, iterations=10)
				else:
					# Try rect seeding from include bbox if available within ROI
					seed_rect = None
					if user_mask_full is not None:
						um = user_mask_full[ry:ry+rh, rx:rx+rw]
						bbox = _bbox_of(um, 1)
						if bbox is not None:
							sx, sy, sw, sh = bbox
							margin = max(2, min(rw, rh) // 20)
							sx = max(0, sx - margin)
							sy = max(0, sy - margin)
							sw = min(rw - sx, sw + 2 * margin)
							sh = min(rh - sy, sh + 2 * margin)
							seed_rect = (sx, sy, max(1, sw), max(1, sh))
					if seed_rect is None:
						# Fallback to inset ROI
						seed_rect = (1, 1, max(1, rw - 2), max(1, rh - 2))
					gc_mask_roi = apply_grabcut(bgr, rect_xywh=seed_rect, init_mask=None, iterations=10)
				# Compose full canvas mask
				gc_mask = np.zeros((h_full, w_full), dtype=np.uint8)
				gc_mask[ry:ry+rh, rx:rx+rw] = gc_mask_roi
			else:
				# No crop: operate on full image
				bgr = bgr_full
				init_mask: Optional[np.ndarray] = None
				use_mask_init = False
				seed_rect_full: Optional[tuple[int, int, int, int]] = None
				if user_mask_full is not None and (user_mask_full != 0).any():
					init_mask = np.zeros((h_full, w_full), dtype=np.uint8)
					# Set user marks
					init_mask[user_mask_full == 1] = 3  # PR_FGD
					init_mask[user_mask_full == 2] = 2  # PR_BGD
					
					# For better GrabCut performance, set a reasonable initial guess
					# If we have include marks, create a bounding box around them and set as PR_FGD
					bbox = _bbox_of(user_mask_full, 1)
					if bbox is not None:
						sx, sy, sw, sh = bbox
						# Expand the bounding box slightly for better initialization
						margin = max(10, min(w_full, h_full) // 20)
						sx = max(0, sx - margin)
						sy = max(0, sy - margin)
						sw = min(w_full - sx, sw + 2 * margin)
						sh = min(h_full - sy, sh + 2 * margin)
						# Set the expanded area as PR_FGD (probably foreground)
						init_mask[sy:sy+sh, sx:sx+sw] = 3
					
					has_prfgd = (init_mask == 3).any()
					has_prbgd = (init_mask == 2).any()
					# Use mask initialization if we have any foreground marks, even without background marks
					use_mask_init = has_prfgd
					if not use_mask_init:
						bbox = _bbox_of(user_mask_full, 1)
						if bbox is not None:
							sx, sy, sw, sh = bbox
							margin = max(2, min(w_full, h_full) // 20)
							sx = max(0, sx - margin)
							sy = max(0, sy - margin)
							sw = min(w_full - sx, sw + 2 * margin)
							sh = min(h_full - sy, sh + 2 * margin)
							seed_rect_full = (sx, sy, max(1, sw), max(1, sh))
				if use_mask_init:
					gc_mask = apply_grabcut(bgr, rect_xywh=None, init_mask=init_mask, iterations=10)
				elif seed_rect_full is not None:
					gc_mask = apply_grabcut(bgr, rect_xywh=seed_rect_full, init_mask=None, iterations=10)
				else:
					QMessageBox.information(self, "Background Removal", "Add Include marks or set a Crop rectangle first.")
					return

			fg01 = ((gc_mask == 1) | (gc_mask == 3)).astype(np.uint8)
			rgba = composite_foreground_over_transparent(bgr_full, fg01)
			self.set_working_image(rgba)
		except Exception as e:  # noqa: BLE001
			QMessageBox.warning(self, "Background Removal", f"GrabCut failed: {e}")
		finally:
			self.btn_run.setEnabled(True)
			QApplication.restoreOverrideCursor()

	def _on_run_rembg(self) -> None:
		# Get working image from app state
		working_image = self._app_state.working_image
		if working_image is None:
			return
		# Note: statusBar() would need to be passed from MainWindow
		self.btn_ai.setEnabled(False)
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			# Convert RGBA to BGR for rembg processing
			bgr_full = working_image[:, :, [2, 1, 0]]  # Convert RGBA to BGR
			h, w = bgr_full.shape[:2]
			rect = self._background_removal_view.get_crop_rect_xywh()
			user_mask = self._background_removal_view.get_user_mask()
			model_id = self.get_selected_rembg_model()

			if rect is not None:
				rx, ry, rw, rh = rect
				rx = max(0, min(rx, w - 1)); ry = max(0, min(ry, h - 1))
				rw = max(1, min(rw, w - rx)); rh = max(1, min(rh, h - ry))
				roi = bgr_full[ry:ry+rh, rx:rx+rw]
				rgba_roi = rembg_remove_bgr_to_rgba(roi, model=model_id, target_hw=(rh, rw))
				# Create full-size output
				rgba_full = np.zeros((h, w, 4), dtype=np.uint8)
				rgba_full[ry:ry+rh, rx:rx+rw] = rgba_roi
			else:
				rgba_full = rembg_remove_bgr_to_rgba(bgr_full, model=model_id, target_hw=(h, w))

			# Create a writable copy for applying user mask constraints
			rgba_working = np.array(rgba_full, copy=True)

			# Apply user mask constraints if any
			if user_mask is not None:
				rgba_working[user_mask == 2, 3] = 0  # Exclude areas
				rgba_working[user_mask == 1, 3] = 255  # Include areas

			# Store the original AI image BEFORE clamping for threshold slider
			self._working_ai_image = rgba_working.copy()

			# Apply transparency clamping to ensure binary alpha values
			rgba_working = clamp_alpha_channel(rgba_working, threshold=128)
			
			# Update working image using BaseStep centralized management
			self.set_working_image(rgba_working)
		except Exception as e:  # noqa: BLE001
			QMessageBox.warning(self, "AI Background Removal", f"rembg failed: {e}")
		finally:
			self.btn_ai.setEnabled(True)
			QApplication.restoreOverrideCursor()

	def _on_open(self):
		super()._on_open()

		# Wire signals
		self.radio_crop.toggled.connect(self._handle_mode_change)
		self.radio_include.toggled.connect(self._handle_mode_change)
		self.radio_exclude.toggled.connect(self._handle_mode_change)
		self.radio_erase.toggled.connect(self._handle_mode_change)
		self.slider_brush.valueChanged.connect(self._background_removal_view.set_brush_size)
		self.slider_opacity_threshold.valueChanged.connect(self._on_opacity_threshold_changed)
		self.btn_undo.clicked.connect(self._background_removal_view.undo)
		self.btn_redo.clicked.connect(self._background_removal_view.redo)
		self.btn_clear.clicked.connect(self._background_removal_view.clear_marks)
		self.btn_run.clicked.connect(self._on_run_grabcut)
		self.btn_ai.clicked.connect(self._on_run_rembg)
		self.btn_apply_crop.clicked.connect(self._background_removal_view.apply_crop)
		self._handle_mode_change()
		self._background_removal_view.setVisible(True)
		self._background_removal_view.raise_()  # Ensure it's on top
		self._background_removal_view.show_overlays()
	
	def _on_close(self):
		super()._on_close()
		self.radio_crop.toggled.disconnect(self._handle_mode_change)
		self.radio_include.toggled.disconnect(self._handle_mode_change)
		self.radio_exclude.toggled.disconnect(self._handle_mode_change)
		self.radio_erase.toggled.disconnect(self._handle_mode_change)
		self.slider_brush.valueChanged.disconnect(self._background_removal_view.set_brush_size)
		self.slider_opacity_threshold.valueChanged.disconnect(self._on_opacity_threshold_changed)
		self.btn_undo.clicked.disconnect(self._background_removal_view.undo)
		self.btn_redo.clicked.disconnect(self._background_removal_view.redo)
		self.btn_clear.clicked.disconnect(self._background_removal_view.clear_marks)
		self._background_removal_view.hide_overlays()
		self._image_view.set_mode("none")
		self._working_ai_image = None
