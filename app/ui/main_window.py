from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QImage
from PySide6.QtWidgets import (
	QDockWidget,
	QFileDialog,
	QMainWindow,
	QMessageBox,
	QToolBar,
	QApplication,
	QTabWidget,
)

from app.ui.image_view import ImageView
from app.ui.bg_tools_panel import BgToolsPanel
from app.ui.color_simplify_panel import ColorSimplifyPanel
from app.utils.qt_image import qimage_to_numpy_bgr, composite_foreground_over_transparent, numpy_rgba_to_qimage
from app.processing.grabcut import apply_grabcut
from app.processing.rembg_infer import rembg_remove_bgr_to_rgba
from app.processing.matting_refine import refine_alpha_portrait
from app.processing.color_simplify import simplify_colors_adaptive, get_color_statistics


class MainWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Image Segmenter & SVG Layout")
		self.resize(1200, 800)

		self._image_view = ImageView(self)
		self.setCentralWidget(self._image_view)

		# Store original AI output for re-processing with different thresholds
		self._original_ai_output: Optional[np.ndarray] = None
		# Store simplified output for color simplification
		self._simplified_output: Optional[np.ndarray] = None

		self._create_actions()
		self._create_menus_and_toolbar()
		self._create_docks()

	def _create_actions(self) -> None:
		self.action_open = QAction("&Open Image…", self)
		self.action_open.setShortcut(QKeySequence.Open)
		self.action_open.triggered.connect(self._on_open_image)

		self.action_exit = QAction("E&xit", self)
		self.action_exit.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Q))
		self.action_exit.triggered.connect(self.close)

		self.action_about = QAction("&About", self)
		self.action_about.triggered.connect(self._on_about)

	def _create_menus_and_toolbar(self) -> None:
		file_menu = self.menuBar().addMenu("&File")
		file_menu.addAction(self.action_open)
		file_menu.addSeparator()
		file_menu.addAction(self.action_exit)

		help_menu = self.menuBar().addMenu("&Help")
		help_menu.addAction(self.action_about)

		toolbar = QToolBar("Main Toolbar", self)
		toolbar.setMovable(True)
		toolbar.addAction(self.action_open)
		self.addToolBar(Qt.TopToolBarArea, toolbar)

	def _create_docks(self) -> None:
		# Create tabbed widget for tools
		self._tab_widget = QTabWidget(self)
		self._tab_widget.setObjectName("ToolsTabWidget")
		self._tab_widget.setTabPosition(QTabWidget.North)
		
		# Background removal tab
		self._bg_panel = BgToolsPanel(self)
		self._tab_widget.addTab(self._bg_panel, "1. Background Removal")
		
		# Color simplification tab
		self._color_panel = ColorSimplifyPanel(self)
		self._tab_widget.addTab(self._color_panel, "2. Color Simplification")
		
		# Create dock widget for the tabbed interface
		self._dock_tools = QDockWidget("Workflow Tools", self)
		self._dock_tools.setObjectName("DockTools")
		self._dock_tools.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self._dock_tools.setWidget(self._tab_widget)
		self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_tools)

		# Wire bg panel
		self._bg_panel.modeChanged.connect(self._image_view.set_mode)
		self._bg_panel.brushSizeChanged.connect(self._image_view.set_brush_size)
		self._bg_panel.undoRequested.connect(self._image_view.undo)
		self._bg_panel.redoRequested.connect(self._image_view.redo)
		self._bg_panel.clearRequested.connect(self._image_view.clear_marks)
		self._bg_panel.previewToggled.connect(self._image_view.set_preview_enabled)
		self._bg_panel.applyCropRequested.connect(self._image_view.apply_crop)
		self._bg_panel.runGrabcutRequested.connect(self._on_run_grabcut)
		self._bg_panel.aiRembgRequested.connect(self._on_run_rembg)
		self._bg_panel.portraitMattingRequested.connect(self._on_refine_portrait)
		self._bg_panel.opacityThresholdChanged.connect(self._on_opacity_threshold_changed)

		# Wire color panel
		self._color_panel.simplifyRequested.connect(self._on_simplify_colors)
		self._color_panel.previewToggled.connect(self._image_view.set_preview_enabled)
		self._color_panel.applyRequested.connect(self._on_apply_simplification)
		
		# Connect tab changes to update workflow guidance
		self._tab_widget.currentChanged.connect(self._on_tab_changed)

	def _on_tab_changed(self, index: int) -> None:
		"""Handle tab changes to provide workflow guidance."""
		if index == 0:  # Background Removal tab
			self.statusBar().showMessage("Step 1: Remove background using AI or manual tools", 3000)
		elif index == 1:  # Color Simplification tab
			if self._original_ai_output is not None:
				self.statusBar().showMessage("Step 2: Simplify colors for laser engraving", 3000)
			else:
				self.statusBar().showMessage("Complete background removal first, then simplify colors", 3000)

	def _update_tab_availability(self) -> None:
		"""Update tab availability based on workflow state."""
		# Enable/disable color simplification tab based on whether background removal is complete
		has_background_removed = self._original_ai_output is not None
		self._tab_widget.setTabEnabled(1, has_background_removed)
		
		# Update tab text to show status
		if has_background_removed:
			self._tab_widget.setTabText(1, "2. Color Simplification ✓")
		else:
			self._tab_widget.setTabText(1, "2. Color Simplification (Complete Step 1 first)")

	def _switch_to_color_simplification(self) -> None:
		"""Switch to the color simplification tab."""
		self._tab_widget.setCurrentIndex(1)

	def _on_open_image(self) -> None:
		start_dir = str(Path.home())
		filters = "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)"
		file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", start_dir, filters)
		if not file_path:
			return

		image = QImage(file_path)
		if image.isNull():
			QMessageBox.warning(self, "Open Image", f"Failed to load image: {file_path}")
			return

		pixmap = QPixmap.fromImage(image)
		self._image_view.set_pixmap(pixmap)
		# Clear any stored outputs when loading a new image
		self._original_ai_output = None
		self._simplified_output = None
		
		# Update color statistics
		bgr = qimage_to_numpy_bgr(image)
		rgb = bgr[:, :, ::-1]
		rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
		stats = get_color_statistics(rgba)
		self._color_panel.update_statistics(stats)
		
		# Update tab availability
		self._update_tab_availability()
		
		self.statusBar().showMessage(f"Loaded {file_path}", 5000)

	def _on_run_grabcut(self) -> None:
		if self._image_view._orig_qimage is None:  # noqa: SLF001 - simple access
			return
		# Show busy indicator
		self.statusBar().showMessage("Removing background…")
		self._bg_panel.btn_run.setEnabled(False)
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			bgr_full = qimage_to_numpy_bgr(self._image_view._orig_qimage)  # noqa: SLF001
			h_full, w_full = bgr_full.shape[:2]
			rect = self._image_view.get_crop_rect_xywh()
			user_mask_full = self._image_view.get_user_mask()

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
				# Validate seeds
				use_mask_init = False
				if init_mask_roi is not None:
					has_prfgd = (init_mask_roi == 3).any()
					has_prbgd = (init_mask_roi == 2).any()
					use_mask_init = has_prfgd and has_prbgd
				if use_mask_init:
					gc_mask_roi = apply_grabcut(bgr, rect_xywh=None, init_mask=init_mask_roi, iterations=5)
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
					gc_mask_roi = apply_grabcut(bgr, rect_xywh=seed_rect, init_mask=None, iterations=5)
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
					init_mask[user_mask_full == 1] = 3
					init_mask[user_mask_full == 2] = 2
					has_prfgd = (init_mask == 3).any()
					has_prbgd = (init_mask == 2).any()
					use_mask_init = has_prfgd and has_prbgd
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
					gc_mask = apply_grabcut(bgr, rect_xywh=None, init_mask=init_mask, iterations=5)
				elif seed_rect_full is not None:
					gc_mask = apply_grabcut(bgr, rect_xywh=seed_rect_full, init_mask=None, iterations=5)
				else:
					QMessageBox.information(self, "Background Removal", "Add Include marks or set a Crop rectangle first.")
					return

			fg01 = ((gc_mask == 1) | (gc_mask == 3)).astype(np.uint8)
			rgba = composite_foreground_over_transparent(bgr_full, fg01)
			qimg = numpy_rgba_to_qimage(rgba)
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			self.statusBar().showMessage("Background removal complete. Switch to Color Simplification tab to continue.", 5000)
			# Update tab availability
			self._update_tab_availability()
			# Optionally auto-switch to color simplification tab
			# self._switch_to_color_simplification()
		except Exception as e:  # noqa: BLE001
			QMessageBox.warning(self, "Background Removal", f"GrabCut failed: {e}")
		finally:
			self._bg_panel.btn_run.setEnabled(True)
			QApplication.restoreOverrideCursor()

	def _on_run_rembg(self) -> None:
		if self._image_view._orig_qimage is None:  # noqa: SLF001
			return
		self.statusBar().showMessage("AI Removing background…")
		self._bg_panel.btn_ai.setEnabled(False)
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			bgr_full = qimage_to_numpy_bgr(self._image_view._orig_qimage)  # noqa: SLF001
			h, w = bgr_full.shape[:2]
			rect = self._image_view.get_crop_rect_xywh()
			user_mask = self._image_view.get_user_mask()
			model_id = self._bg_panel.get_selected_rembg_model()

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

			# Store original AI output for re-processing (ensure it's writable)
			self._original_ai_output = np.array(rgba_full, copy=True)

			# Create a writable copy for applying user mask constraints
			rgba_working = np.array(rgba_full, copy=True)

			# Apply user mask constraints if any
			if user_mask is not None:
				rgba_working[user_mask == 2, 3] = 0  # Exclude areas
				rgba_working[user_mask == 1, 3] = 255  # Include areas

			# Convert to QImage and set as preview
			qimg = numpy_rgba_to_qimage(rgba_working)
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			self.statusBar().showMessage("AI background removal complete. Switch to Color Simplification tab to continue.", 5000)
			# Update tab availability
			self._update_tab_availability()
			# Optionally auto-switch to color simplification tab
			# self._switch_to_color_simplification()
		except Exception as e:  # noqa: BLE001
			QMessageBox.warning(self, "AI Background Removal", f"rembg failed: {e}")
		finally:
			self._bg_panel.btn_ai.setEnabled(True)
			QApplication.restoreOverrideCursor()

	def _on_refine_portrait(self) -> None:
		if self._image_view._orig_qimage is None:  # noqa: SLF001
			return
		self.statusBar().showMessage("Refining portrait matte…")
		QApplication.setOverrideCursor(Qt.WaitCursor)
		try:
			bgr_full = qimage_to_numpy_bgr(self._image_view._orig_qimage)  # noqa: SLF001
			rgb_full = bgr_full[:, :, ::-1]
			h, w = rgb_full.shape[:2]
			user_mask = self._image_view.get_user_mask()
			# Use current preview alpha if available; else run rembg general as init
			alpha_init: Optional[np.ndarray] = None
			if self._image_view._preview_pixmap is not None:  # noqa: SLF001
				qimg = self._image_view._preview_pixmap.toImage()  # noqa: SLF001
				qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
				ptr = qimg.constBits()
				arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
				alpha_init = arr[:, :, 3].copy()
			else:
				rgba = rembg_remove_bgr_to_rgba(bgr_full, model="isnet-general-use", target_hw=(h, w))
				alpha_init = rgba[:, :, 3].copy()
			alpha_refined = refine_alpha_portrait(rgb_full, alpha_init, user_mask)
			rgba_full = np.dstack([rgb_full, alpha_refined])
			# Store the refined output as the new original AI output (ensure it's writable)
			self._original_ai_output = np.array(rgba_full, copy=True)
			qimg_out = numpy_rgba_to_qimage(rgba_full)
			self._image_view.set_preview_image(qimg_out)
			self._image_view.set_preview_enabled(True)
			self.statusBar().showMessage("Portrait matte refined. Switch to Color Simplification tab to continue.", 5000)
			# Update tab availability
			self._update_tab_availability()
			# Optionally auto-switch to color simplification tab
			# self._switch_to_color_simplification()
		except Exception as e:  # noqa: BLE001
			# Check if it's a convergence error and provide a helpful message
			if "converge" in str(e).lower():
				QMessageBox.information(
					self, 
					"Portrait Matting", 
					"Advanced matting failed to converge. Using simplified refinement instead. "
					"This is normal for complex images or when the initial alpha mask is unclear."
				)
			else:
				QMessageBox.warning(self, "Portrait Matting", f"Refine failed: {e}")
		finally:
			QApplication.restoreOverrideCursor()

	def _on_about(self) -> None:
		QMessageBox.information(
			self,
			"About",
			"""
			Image Segmenter & SVG Layout
			
			A tool to simplify images, segment by color, arrange segments for minimal layout, and export SVG for laser engraving.
			""".strip(),
		)

	def _on_opacity_threshold_changed(self, threshold: int) -> None:
		"""Handle opacity threshold changes by re-processing the AI output."""
		if self._original_ai_output is not None:
			# Update the image view's threshold
			self._image_view.set_opacity_threshold(threshold)
			# Re-process the original AI output with new threshold
			# Ensure we have a writable copy
			rgba_working = np.array(self._original_ai_output, copy=True)
			qimg = numpy_rgba_to_qimage(rgba_working)
			self._image_view.set_preview_image(qimg)

	def _on_simplify_colors(self) -> None:
		"""Handle color simplification request.
		
		Note: Always uses the original background-removed image as input to prevent
		cascading quality loss from multiple simplifications.
		"""
		# Always use the original AI output as the base for simplification
		if self._original_ai_output is not None:
			source_rgba = self._original_ai_output.copy()
		elif self._image_view._orig_qimage is not None:  # noqa: SLF001
			# Fallback to original image if no AI output available
			bgr = qimage_to_numpy_bgr(self._image_view._orig_qimage)  # noqa: SLF001
			rgb = bgr[:, :, ::-1]
			# Create RGBA with full alpha
			source_rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
		else:
			QMessageBox.warning(self, "Color Simplification", "No image available for simplification.")
			return
		
		if source_rgba is None:
			return
		
		self.statusBar().showMessage("Simplifying colors…")
		QApplication.setOverrideCursor(Qt.WaitCursor)
		
		try:
			# Get parameters from the color panel
			num_colors = self._color_panel.get_num_colors()
			algorithm = self._color_panel.get_algorithm()
			preserve_alpha = self._color_panel.get_preserve_alpha()
			
			# Perform color simplification
			algorithm = self._color_panel.get_algorithm()
			simplified_rgba, palette = simplify_colors_adaptive(
				source_rgba, 
				num_colors, 
				preserve_alpha,
				algorithm
			)
			
			# Store the simplified output
			self._simplified_output = simplified_rgba.copy()
			
			# Update statistics with palette information
			stats = get_color_statistics(simplified_rgba)
			stats['palette'] = palette
			self._color_panel.update_statistics(stats)
			
			# Show preview
			qimg = numpy_rgba_to_qimage(simplified_rgba)
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			
			algorithm_name = {
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
			self.statusBar().showMessage(f"Color simplification complete using {algorithm_name}. Reduced from original to {num_colors} colors.", 3000)
			
		except Exception as e:  # noqa: BLE001
			QMessageBox.warning(self, "Color Simplification", f"Color simplification failed: {e}")
		finally:
			QApplication.restoreOverrideCursor()



	def _on_apply_simplification(self) -> None:
		"""Apply the color simplification to the current image."""
		if self._simplified_output is None:
			QMessageBox.warning(self, "Apply Simplification", "No simplified image to apply.")
			return
		
		# Convert the simplified output back to the original image format
		if self._image_view._orig_qimage is not None:  # noqa: SLF001
			# Create a new QImage from the simplified RGBA
			qimg = numpy_rgba_to_qimage(self._simplified_output)
			pixmap = QPixmap.fromImage(qimg)
			
			# Set as the new base image
			self._image_view.set_pixmap(pixmap)
			
			# Clear the stored outputs since we're starting fresh
			self._original_ai_output = None
			self._simplified_output = None
			
			# Update statistics
			stats = get_color_statistics(self._simplified_output) if self._simplified_output is not None else None
			self._color_panel.update_statistics(stats)
			
			# Update tab availability since we're starting fresh
			self._update_tab_availability()
			
			self.statusBar().showMessage("Simplification applied to image. Workflow reset - start with background removal.", 5000)
		else:
			QMessageBox.warning(self, "Apply Simplification", "No original image available.")
