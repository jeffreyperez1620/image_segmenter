from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QImage, QColor
from PySide6.QtWidgets import (
	QDockWidget,
	QFileDialog,
	QMainWindow,
	QMessageBox,
	QToolBar,
	QApplication,
	QTabWidget,
	QDialog,
	QDialogButtonBox,
	QGridLayout,
	QLabel,
	QScrollArea,
	QWidget,
	QVBoxLayout,
)

from ui.image_view import ImageView
from ui.bg_tools_panel import BgToolsPanel
from ui.color_processing_panel import ColorProcessingPanel, ColorSwatch
from ui.region_cleanup_panel import RegionCleanupPanel
from ui.progress_dialog import ProgressDialog
from utils.qt_image import qimage_to_numpy_bgr, composite_foreground_over_transparent, numpy_rgba_to_qimage
from processing.grabcut import apply_grabcut
from processing.rembg_infer import rembg_remove_bgr_to_rgba
from processing.matting_refine import refine_alpha_portrait
from processing.color_simplify import simplify_colors_adaptive, get_color_statistics
from processing.region_cleanup import analyze_regions, merge_small_regions, flood_fill_region, get_region_boundaries


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
		# Store cleaned output for region cleanup
		self._cleaned_output: Optional[np.ndarray] = None
		# Track if color processing has been applied
		self._color_processing_applied: bool = False
		# Store original image and saved preview for Show Original toggle
		self._original_image: Optional[QPixmap] = None
		self._saved_preview_pixmap: Optional[QPixmap] = None
		# Track previous tab index for eyedropper deactivation
		self._previous_tab_index: int = 0

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
		
		# Color processing tab (combines simplification and custom palette)
		self._color_processing_panel = ColorProcessingPanel(self)
		self._tab_widget.addTab(self._color_processing_panel, "2. Color Processing")
		
		# Region cleanup tab
		self._region_cleanup_panel = RegionCleanupPanel(self)
		self._tab_widget.addTab(self._region_cleanup_panel, "3. Region Cleanup")
		
		# Create dock widget for the tabbed interface
		self._dock_tools = QDockWidget("Workflow Tools", self)
		self._dock_tools.setObjectName("DockTools")
		self._dock_tools.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self._dock_tools.setWidget(self._tab_widget)
		self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_tools)
		
		# Connect tab change signal to populate flood fill palette when entering Step 3
		self._tab_widget.currentChanged.connect(self._on_tab_changed)

		# Wire bg panel
		self._bg_panel.modeChanged.connect(self._image_view.set_mode)
		self._bg_panel.brushSizeChanged.connect(self._image_view.set_brush_size)
		self._bg_panel.undoRequested.connect(self._image_view.undo)
		self._bg_panel.redoRequested.connect(self._image_view.redo)
		self._bg_panel.clearRequested.connect(self._image_view.clear_marks)
		self._bg_panel.previewToggled.connect(self._on_show_original_toggled)
		self._bg_panel.applyCropRequested.connect(self._image_view.apply_crop)
		self._bg_panel.runGrabcutRequested.connect(self._on_run_grabcut)
		self._bg_panel.aiRembgRequested.connect(self._on_run_rembg)
		self._bg_panel.portraitMattingRequested.connect(self._on_refine_portrait)
		self._bg_panel.opacityThresholdChanged.connect(self._on_opacity_threshold_changed)
		# self._bg_panel.resetRequested.connect(self._on_reset_to_original)  # Signal doesn't exist

		# Wire color processing panel
		self._color_processing_panel.algorithmProcessRequested.connect(self._on_algorithm_process)
		self._color_processing_panel.paletteProcessRequested.connect(self._on_palette_process)
		self._color_processing_panel.previewToggled.connect(self._image_view.set_preview_enabled)
		self._color_processing_panel.applyRequested.connect(self._on_apply_color_processing)
		self._color_processing_panel.eyedropperRequested.connect(self._on_start_eyedropper)
		self._color_processing_panel.eyedropperCancelled.connect(self._on_eyedropper_cancelled)
		
		# Wire image view color picking
		self._image_view.colorPicked.connect(self._on_color_picked)
		self._image_view.colorPreview.connect(self._color_processing_panel.update_color_preview)
		self._image_view.eyedropperCancelled.connect(self._on_eyedropper_cancelled)
		
		# Wire region cleanup panel
		self._region_cleanup_panel.cleanupRequested.connect(self._on_cleanup_regions)
		self._region_cleanup_panel.previewToggled.connect(self._image_view.set_preview_enabled)
		self._region_cleanup_panel.applyRequested.connect(self._on_apply_region_cleanup)
		self._region_cleanup_panel.floodFillRequested.connect(self._on_start_flood_fill)
		self._region_cleanup_panel.smoothingRequested.connect(self._on_smooth_regions)
		self._region_cleanup_panel.regionBoundariesToggled.connect(self._on_region_boundaries_toggled)
		self._region_cleanup_panel.saveRequested.connect(self._on_save_working_image)
		
		# Wire image view flood fill signal
		self._image_view.floodFillRequested.connect(self._on_flood_fill_requested)
		
		# Connect tab changes to update workflow guidance
		self._tab_widget.currentChanged.connect(self._on_tab_changed)

	def _on_tab_changed(self, index: int) -> None:
		"""Handle tab changes to provide workflow guidance."""
		# Disable eyedropper when leaving color processing tab
		if index != 1:  # Not on color processing tab
			self._image_view.set_mode("include")  # Reset to default mode
			self._color_processing_panel.reset_eyedropper_state()
		
		if index == 0:  # Background Removal tab
			self.statusBar().showMessage("Step 1: Remove background using AI or manual tools", 3000)
		elif index == 1:  # Color Processing tab
			if self._original_ai_output is not None:
				self.statusBar().showMessage("Step 2: Process colors using algorithms or create custom palette", 3000)
			else:
				self.statusBar().showMessage("Complete background removal first, then process colors", 3000)
		elif index == 2:  # Region Cleanup tab
			if self._simplified_output is not None or self._color_processing_applied:
				self.statusBar().showMessage("Step 3: Clean up small regions for laser engraving", 3000)
			else:
				self.statusBar().showMessage("Complete color processing first", 3000)

	def _update_tab_availability(self) -> None:
		"""Update tab availability based on workflow state."""
		# Enable/disable tabs based on workflow state
		has_background_removed = self._original_ai_output is not None
		has_color_processed = self._simplified_output is not None or self._color_processing_applied
		
		# Color processing tab
		self._tab_widget.setTabEnabled(1, has_background_removed)
		if has_background_removed:
			self._tab_widget.setTabText(1, "2. Color Processing ✓")
		else:
			self._tab_widget.setTabText(1, "2. Color Processing (Complete Step 1 first)")
		
		# Region cleanup tab
		self._tab_widget.setTabEnabled(2, has_color_processed)
		if has_color_processed:
			self._tab_widget.setTabText(2, "3. Region Cleanup ✓")
		else:
			self._tab_widget.setTabText(2, "3. Region Cleanup (Complete Step 2 first)")

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
		# Store the original image for reset functionality
		self._original_image = pixmap
		# Clear any stored outputs when loading a new image
		self._original_ai_output = None
		self._simplified_output = None
		self._cleaned_output = None
		self._color_processing_applied = False
		
		# Enable background tools controls now that an image is loaded
		self._bg_panel.set_controls_enabled(True)
		
		# Update color statistics
		bgr = qimage_to_numpy_bgr(image)
		rgb = bgr[:, :, ::-1]
		rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
		stats = get_color_statistics(rgba)
		self._color_processing_panel.update_statistics(stats)
		
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
			qimg = numpy_rgba_to_qimage(rgba)
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			# Update show original checkbox to unchecked since we're showing preview
			self._bg_panel.chk_show_original.setChecked(False)
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
			# Update show original checkbox to unchecked since we're showing preview
			self._bg_panel.chk_show_original.setChecked(False)
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
			# Update show original checkbox to unchecked since we're showing preview
			self._bg_panel.chk_show_original.setChecked(False)
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

	def _on_show_original_toggled(self, show_original: bool) -> None:
		"""Handle show original checkbox toggle."""
		if show_original and self._original_image is not None:
			# Show original image
			self._image_view.set_pixmap(self._original_image)
			self._image_view.set_preview_enabled(False)
		else:
			# Show current working image (preview if available)
			if self._image_view.get_preview_image() is not None:
				self._image_view.set_preview_enabled(True)
			else:
				self._image_view.set_preview_enabled(False)

	def _on_reset_to_original(self) -> None:
		"""Reset the working image to the original image."""
		if self._original_image is not None:
			# Reset to original image
			self._image_view.set_pixmap(self._original_image)
			# Clear any preview
			self._image_view.set_preview_image(None)
			# Clear any working data
			self._original_ai_output = None
			# Clear any marks/masks
			self._image_view.clear_marks()
			# Update show original checkbox to unchecked since we're showing original
			# self._bg_panel.chk_show_original.setChecked(False)  # Checkbox doesn't exist
			# Update status
			self.statusBar().showMessage("Reset to original image")
		else:
			QMessageBox.information(self, "Reset", "No original image to reset to")

	def _on_algorithm_process(self) -> None:
		"""Handle algorithm-based color processing request."""
		# Always use the original AI output as the base for processing
		if self._original_ai_output is not None:
			source_rgba = self._original_ai_output.copy()
		elif self._image_view._orig_qimage is not None:  # noqa: SLF001
			# Fallback to original image if no AI output available
			bgr = qimage_to_numpy_bgr(self._image_view._orig_qimage)  # noqa: SLF001
			rgb = bgr[:, :, ::-1]
			# Create RGBA with full alpha
			source_rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
		else:
			QMessageBox.warning(self, "Algorithm Processing", "No image available for processing.")
			return
		
		if source_rgba is None:
			return
		
		# Get the selected algorithm
		algorithm = self._color_processing_panel.get_algorithm()
		self._process_algorithm_based(source_rgba, algorithm)

	def _on_palette_process(self) -> None:
		"""Handle custom palette color processing request."""
		# Always use the original AI output as the base for processing
		if self._original_ai_output is not None:
			source_rgba = self._original_ai_output.copy()
		elif self._image_view._orig_qimage is not None:  # noqa: SLF001
			# Fallback to original image if no AI output available
			bgr = qimage_to_numpy_bgr(self._image_view._orig_qimage)  # noqa: SLF001
			rgb = bgr[:, :, ::-1]
			# Create RGBA with full alpha
			source_rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
		else:
			QMessageBox.warning(self, "Custom Palette Processing", "No image available for processing.")
			return
		
		if source_rgba is None:
			return
		
		self._process_custom_palette(source_rgba)

	def _process_algorithm_based(self, source_rgba: np.ndarray, algorithm: str) -> None:
		"""Process colors using algorithm-based simplification."""
		self.statusBar().showMessage("Processing colors with algorithm...")
		QApplication.setOverrideCursor(Qt.WaitCursor)
		
		try:
			# Get parameters from the color processing panel
			num_colors = self._color_processing_panel.get_num_colors()
			preserve_alpha = self._color_processing_panel.get_preserve_alpha()
			
			# Perform color simplification
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
			self._color_processing_panel.update_statistics(stats)
			
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
			self.statusBar().showMessage(f"Color processing complete using {algorithm_name}. Reduced to {num_colors} colors.", 3000)
			
		except Exception as e:  # noqa: BLE001
			QMessageBox.warning(self, "Color Processing", f"Color processing failed: {e}")
		finally:
			QApplication.restoreOverrideCursor()

	def _process_custom_palette(self, source_rgba: np.ndarray) -> None:
		"""Process colors using custom palette."""
		self.statusBar().showMessage("Processing colors with custom palette...")
		QApplication.setOverrideCursor(Qt.WaitCursor)
		
		try:
			# Get parameters from the color processing panel
			custom_palette = self._color_processing_panel.get_palette()
			distance_metric = self._color_processing_panel.get_distance_metric()
			preserve_alpha = self._color_processing_panel.get_preserve_alpha()
			
			# Perform custom palette simplification
			from processing.color_simplify import simplify_colors_custom_palette
			simplified_rgba, palette = simplify_colors_custom_palette(
				source_rgba, 
				custom_palette, 
				preserve_alpha,
				distance_metric
			)
			
			# Store the simplified output
			self._simplified_output = simplified_rgba.copy()
			
			# Update statistics with palette information
			stats = get_color_statistics(simplified_rgba)
			stats['palette'] = palette
			self._color_processing_panel.update_statistics(stats)
			
			# Show preview
			qimg = numpy_rgba_to_qimage(simplified_rgba)
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			
			metric_name = {
				"lab": "LAB (Perceptual)",
				"rgb": "RGB (Euclidean)",
				"hsv": "HSV"
			}.get(distance_metric, distance_metric)
			
			self.statusBar().showMessage(f"Custom palette processing complete using {metric_name}. Applied {len(palette)} colors.", 3000)
			
		except Exception as e:  # noqa: BLE001
			QMessageBox.warning(self, "Custom Palette", f"Custom palette processing failed: {e}")
		finally:
			QApplication.restoreOverrideCursor()



	def _on_apply_color_processing(self) -> None:
		"""Apply the color processing to the current image."""
		# Deactivate eyedropper if it's active
		self._on_eyedropper_cancelled()
		
		if self._simplified_output is None:
			QMessageBox.warning(self, "Apply Color Processing", "No processed image to apply.")
			return
		
		# Convert the processed output back to the original image format
		if self._image_view._orig_qimage is not None:  # noqa: SLF001
			# Create a new QImage from the processed RGBA
			qimg = numpy_rgba_to_qimage(self._simplified_output)
			pixmap = QPixmap.fromImage(qimg)
			
			# Set as the new base image
			self._image_view.set_pixmap(pixmap)
			
			# Update the original AI output to the processed result
			# This allows the next step (Region Cleanup) to work on the processed image
			self._original_ai_output = self._simplified_output.copy()
			
			# Clear the processed output since it's now the base
			self._simplified_output = None
			self._cleaned_output = None
			
			# Mark that color processing has been applied
			self._color_processing_applied = True
			
			# Update statistics with the new base image
			stats = get_color_statistics(self._original_ai_output)
			self._color_processing_panel.update_statistics(stats)
			
			# Update tab availability to enable the next step
			self._update_tab_availability()
			
			self.statusBar().showMessage("Color processing applied to image. Region Cleanup step is now available.", 5000)
		else:
			QMessageBox.warning(self, "Apply Color Processing", "No original image available.")

	def _on_start_eyedropper(self) -> None:
		"""Start the eyedropper tool."""
		if self._image_view._orig_qimage is None:  # noqa: SLF001
			QMessageBox.warning(self, "Eyedropper", "No image loaded. Please load an image first.")
			return
		
		# Set the image view to eyedropper mode
		self._image_view.set_mode("eyedropper")
		self.statusBar().showMessage("Eyedropper active - click on the image to pick a color", 3000)

	def _on_color_picked(self, color: QColor) -> None:
		"""Handle color picked from the image."""
		# Add the color to the custom palette
		self._color_processing_panel.add_color_from_image(color)
		
		# Keep eyedropper mode active for multiple picks
		# Don't switch back to normal mode
		
		# Show feedback
		self.statusBar().showMessage(f"Color picked: RGB({color.red()}, {color.green()}, {color.blue()}) - Click to pick more colors or press ESC to exit", 3000)

	def _on_eyedropper_cancelled(self) -> None:
		"""Handle eyedropper mode cancellation."""
		# Reset eyedropper button state
		self._color_processing_panel.reset_eyedropper_state()
		# Switch back to normal mode
		self._image_view.set_mode("include")
		self.statusBar().showMessage("Eyedropper cancelled", 2000)

	def _on_cleanup_regions(self) -> None:
		"""Handle region cleanup request."""
		# Check if we have a processed image available (either from preview or applied processing)
		if self._simplified_output is None and not self._color_processing_applied:
			QMessageBox.warning(self, "Region Cleanup", "No processed image available. Complete color processing first.")
			return
		
		# Create progress dialog
		progress_dialog = ProgressDialog("Region Cleanup", self)
		progress_dialog.show()
		
		# Create progress callback
		def progress_callback(current, total, message):
			progress_dialog.update_progress(current, total, message)
			return not progress_dialog.is_cancelled()
		
		try:
			# Get the minimum region size threshold and auto-merge settings
			min_size = self._region_cleanup_panel.get_min_region_size()
			auto_merge_threshold = self._region_cleanup_panel.get_auto_merge_threshold()
			
			# Create merge callback for user interaction
			def merge_callback(small_color, neighbor_colors, image_data=None, bbox=None):
				# Check if user cancelled during progress
				if progress_dialog.is_cancelled():
					return None
				return self._region_cleanup_panel.show_merge_dialog(small_color, neighbor_colors, image_data, bbox)
			
			# Determine which image to use for region cleanup
			if self._simplified_output is not None:
				# Use the preview image if available
				source_rgba = self._simplified_output
			else:
				# Use the applied color processing result
				source_rgba = self._original_ai_output
			
			# Perform region cleanup with hybrid scoring and progress tracking
			cleaned_rgba = merge_small_regions(
				source_rgba, 
				min_size, 
				merge_callback,
				auto_merge_threshold=auto_merge_threshold,
				progress_callback=progress_callback
			)
			
			# Check if user cancelled the operation
			if cleaned_rgba is None or progress_dialog.is_cancelled():
				progress_dialog.close()
				self.statusBar().showMessage("Region cleanup cancelled by user.", 3000)
				return
			
			# Store the cleaned output
			self._cleaned_output = cleaned_rgba.copy()
			
			# Update region statistics
			stats = analyze_regions(cleaned_rgba, min_size)
			self._region_cleanup_panel.update_region_statistics(stats)
			
			# Show preview - ensure array is contiguous
			cleaned_rgba_contiguous = np.ascontiguousarray(cleaned_rgba)
			qimg = numpy_rgba_to_qimage(cleaned_rgba_contiguous)
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			
			# Update region boundaries if they are currently enabled
			if self._region_cleanup_panel.get_show_region_overlay():
				boundaries_data = self._generate_region_boundaries(cleaned_rgba)
				self._image_view.set_region_boundaries_data(boundaries_data)
			
			progress_dialog.close()
			self.statusBar().showMessage(f"Region cleanup complete. Merged regions smaller than {min_size} pixels.", 3000)
			
		except Exception as e:  # noqa: BLE001
			progress_dialog.close()
			QMessageBox.warning(self, "Region Cleanup", f"Region cleanup failed: {e}")

	def _on_region_boundaries_toggled(self, enabled: bool) -> None:
		"""Handle region boundaries overlay toggle."""
		self._image_view.set_region_boundaries_enabled(enabled)
		
		# If enabling, generate and set the boundaries data
		if enabled:
			# Determine which image to use for boundaries (prioritize cleaned output)
			if self._cleaned_output is not None:
				source_rgba = self._cleaned_output
			elif self._simplified_output is not None:
				source_rgba = self._simplified_output
			elif self._original_ai_output is not None:
				source_rgba = self._original_ai_output
			else:
				# No processed image available
				return
			
			# Generate region boundaries
			boundaries_data = self._generate_region_boundaries(source_rgba)
			self._image_view.set_region_boundaries_data(boundaries_data)
		else:
			self._image_view.set_region_boundaries_data(None)
	
	def _generate_region_boundaries(self, rgba: np.ndarray) -> np.ndarray:
		"""Generate region boundaries overlay data."""
		from processing.region_cleanup import get_region_boundaries
		return get_region_boundaries(rgba)
	
	def _on_apply_region_cleanup(self) -> None:
		"""Apply the region cleanup to the current image."""
		if not hasattr(self, '_cleaned_output') or self._cleaned_output is None:
			QMessageBox.warning(self, "Apply Region Cleanup", "No cleaned image to apply.")
			return
		
		# Convert the cleaned output back to the original image format
		if self._image_view._orig_qimage is not None:  # noqa: SLF001
			# Create a new QImage from the cleaned RGBA
			qimg = numpy_rgba_to_qimage(self._cleaned_output)
			pixmap = QPixmap.fromImage(qimg)
			
			# Set as the new base image
			self._image_view.set_pixmap(pixmap)
			
			# Clear the stored outputs since we're starting fresh
			self._original_ai_output = None
			self._simplified_output = None
			self._cleaned_output = None
			
			# Update tab availability since we're starting fresh
			self._update_tab_availability()
			
			self.statusBar().showMessage("Region cleanup applied to image. Workflow reset - start with background removal.", 5000)
		else:
			QMessageBox.warning(self, "Apply Region Cleanup", "No original image available.")

	def _on_start_flood_fill(self) -> None:
		"""Start the flood fill tool."""
		# Determine which image to use for flood fill (same logic as region cleanup)
		if self._simplified_output is not None:
			# Use the preview image if available
			source_rgba = self._simplified_output
		elif self._original_ai_output is not None:
			# Use the applied color processing result
			source_rgba = self._original_ai_output
		else:
			QMessageBox.warning(self, "Flood Fill", "No processed image available. Complete color processing first.")
			return
		
		# Check if flood fill palette is available (should be populated when entering Step 3)
		selected_color = self._region_cleanup_panel.get_selected_flood_fill_color()
		if selected_color is None:
			QMessageBox.warning(self, "Flood Fill", "No colors available in palette. Complete color processing first and return to this tab.")
			return
		
		# Set the image view to flood fill mode
		self._image_view.set_mode("flood_fill")
		self.statusBar().showMessage("Flood fill active - select a color and click on a region to fill it", 3000)
	
	def _on_flood_fill_requested(self, x: int, y: int) -> None:
		"""Handle flood fill request at the specified position."""
		# Get the selected color from the region cleanup panel
		selected_color = self._region_cleanup_panel.get_selected_flood_fill_color()
		if selected_color is None:
			QMessageBox.warning(self, "Flood Fill", "No color selected. Please select a color from the palette first.")
			return
		
		# Determine which image to use for flood fill
		if self._simplified_output is not None:
			source_rgba = self._simplified_output
		elif self._original_ai_output is not None:
			source_rgba = self._original_ai_output
		else:
			QMessageBox.warning(self, "Flood Fill", "No processed image available.")
			return
		
		# Perform flood fill
		try:
			fill_color = (selected_color.red(), selected_color.green(), selected_color.blue())
			filled_rgba = flood_fill_region(source_rgba, (x, y), fill_color)
			
			# Update the simplified output with the filled result
			self._simplified_output = filled_rgba.copy()
			
			# Show the result as preview
			qimg = numpy_rgba_to_qimage(filled_rgba)
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			
			# Update region boundaries if they are currently enabled
			if self._region_cleanup_panel.get_show_region_overlay():
				boundaries_data = self._generate_region_boundaries(filled_rgba)
				self._image_view.set_region_boundaries_data(boundaries_data)
			
			self.statusBar().showMessage(f"Flood fill applied with color RGB({fill_color[0]}, {fill_color[1]}, {fill_color[2]})", 3000)
			
		except Exception as e:
			QMessageBox.warning(self, "Flood Fill", f"Flood fill failed: {e}")
	
	def _show_flood_fill_dialog(self, colors: List[QColor]) -> Optional[QColor]:
		"""Show dialog for selecting flood fill color."""
		dialog = QDialog(self)
		dialog.setWindowTitle("Select Flood Fill Color")
		dialog.setModal(True)
		dialog.resize(400, 300)
		
		layout = QVBoxLayout(dialog)
		
		# Instructions
		instructions = QLabel("Click on a color to use for flood fill:")
		layout.addWidget(instructions)
		
		# Create scrollable grid of color swatches
		scroll_area = QScrollArea()
		scroll_widget = QWidget()
		scroll_layout = QGridLayout(scroll_widget)
		
		cols = 6
		dialog.selected_color = None  # Initialize selected color
		
		for i, color in enumerate(colors):
			swatch = ColorSwatch(color, -1, selectable=True)
			# Use a proper lambda that captures the color correctly
			swatch.colorSelected.connect(lambda c, col=color: self._on_color_selected_in_dialog(dialog, col))
			row = i // cols
			col = i % cols
			scroll_layout.addWidget(swatch, row, col)
		
		scroll_area.setWidget(scroll_widget)
		layout.addWidget(scroll_area)
		
		# Buttons
		button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		ok_button = button_box.button(QDialogButtonBox.Ok)
		ok_button.setEnabled(False)  # Initially disabled
		button_box.accepted.connect(dialog.accept)
		button_box.rejected.connect(dialog.reject)
		layout.addWidget(button_box)
		
		# Store reference to OK button for enabling/disabling
		dialog.ok_button = ok_button
		
		# Show dialog
		if dialog.exec() == QDialog.Accepted:
			return dialog.selected_color
		return None
	
	def _on_color_selected_in_dialog(self, dialog, color: QColor) -> None:
		"""Handle color selection in flood fill dialog."""
		dialog.selected_color = color
		dialog.ok_button.setEnabled(True)
	
	def _on_show_original_toggled(self, show_original: bool) -> None:
		"""Handle Show Original checkbox toggle."""
		if show_original:
			# Show the original image - temporarily save preview if it exists
			if self._image_view._preview_pixmap is not None:
				self._saved_preview_pixmap = self._image_view._preview_pixmap
			else:
				self._saved_preview_pixmap = None
			
			if self._original_image is not None:
				self._image_view.set_pixmap(self._original_image)
				self._image_view.set_preview_enabled(False)
		else:
			# Show the working image (preview if available, otherwise original)
			if hasattr(self, '_saved_preview_pixmap') and self._saved_preview_pixmap is not None:
				# Restore the saved preview image without clearing it
				if self._image_view._pix_item is not None:
					self._image_view._pix_item.setPixmap(self._saved_preview_pixmap)
					# Restore the preview state without clearing it
					self._image_view._preview_pixmap = self._saved_preview_pixmap
					self._image_view._preview_enabled = False
					self._image_view._view.viewport().update()
			else:
				# No preview available, show original
				if self._original_image is not None:
					self._image_view.set_pixmap(self._original_image)
					self._image_view.set_preview_enabled(False)
	
	def _on_tab_changed(self, index: int) -> None:
		"""Handle tab change to populate flood fill palette when entering Step 3."""
		# Deactivate eyedropper when leaving Step 2 (Color Processing)
		if hasattr(self, '_previous_tab_index') and self._previous_tab_index == 1:  # Leaving Step 2
			self._on_eyedropper_cancelled()
		
		if index == 2:  # Step 3: Region Cleanup tab
			# Get available colors from the current palette
			palette_colors = self._color_processing_panel.get_palette()
			if palette_colors is not None and len(palette_colors) > 0:
				# Convert palette colors to QColor list
				colors = []
				for color_array in palette_colors:
					if len(color_array) >= 3:
						color = QColor(int(color_array[0]), int(color_array[1]), int(color_array[2]))
						colors.append(color)
				
				if colors:
					# Update the flood fill palette in the region cleanup panel
					self._region_cleanup_panel.update_flood_fill_palette(colors)
		
		# Store the current tab index for next time
		self._previous_tab_index = index
	
	def _on_cleanup_regions(self) -> None:
		"""Handle region cleanup request."""
		# Check if we have a processed image available (either from preview or applied processing)
		if self._simplified_output is None and not self._color_processing_applied:
			QMessageBox.warning(self, "Region Cleanup", "No processed image available. Complete color processing first.")
			return
		
		# Get parameters from the panel
		min_size = self._region_cleanup_panel.get_min_region_size()
		auto_merge_threshold = self._region_cleanup_panel.get_auto_merge_threshold()
		connectivity = self._region_cleanup_panel.get_connectivity()
		
		# Use the appropriate input image
		if self._simplified_output is not None:
			input_image = self._simplified_output
		else:
			input_image = self._original_ai_output
		
		if input_image is None:
			QMessageBox.warning(self, "Region Cleanup", "No image available for processing.")
			return
		
		# Show progress dialog
		progress_dialog = ProgressDialog("Region Cleanup", self)
		progress_dialog.show()
		QApplication.processEvents()
		
		try:
			# Perform region cleanup
			from processing.region_cleanup import merge_small_regions
			
			def progress_callback(current: int, total: int, message: str) -> None:
				progress_dialog.update_progress(current, total, message)
				QApplication.processEvents()
			
			cleaned_output = merge_small_regions(
				input_image,
				min_size,
				merge_callback=None,  # Auto-merge for now
				auto_merge_threshold=auto_merge_threshold,
				progress_callback=progress_callback,
				connectivity=connectivity
			)
			
			if cleaned_output is not None:
				self._cleaned_output = cleaned_output
				
				# Convert to QImage for preview
				from utils.qt_image import numpy_rgba_to_qimage
				qimg = numpy_rgba_to_qimage(cleaned_output)
				
				# Set as preview
				self._image_view.set_preview_image(qimg)
				self._image_view.set_preview_enabled(True)
				
				# Update statistics
				from app.processing.region_cleanup import analyze_regions
				stats = analyze_regions(cleaned_output, min_size)
				self._region_cleanup_panel.update_region_statistics(stats)
				
				self.statusBar().showMessage("Region cleanup completed. Use 'Apply Cleanup' to make changes permanent.", 5000)
			else:
				QMessageBox.information(self, "Region Cleanup", "Region cleanup was cancelled.")
		
		except Exception as e:
			QMessageBox.critical(self, "Region Cleanup Error", f"An error occurred during region cleanup:\n{str(e)}")
		
		finally:
			progress_dialog.close()
	
	def _on_apply_region_cleanup(self) -> None:
		"""Apply the region cleanup to the current image."""
		if self._cleaned_output is None:
			QMessageBox.warning(self, "Apply Region Cleanup", "No cleaned image to apply.")
			return
		
		# Convert the cleaned output back to the original image format
		if self._image_view._orig_qimage is not None:  # noqa: SLF001
			# Create a new QImage from the cleaned RGBA
			from utils.qt_image import numpy_rgba_to_qimage
			qimg = numpy_rgba_to_qimage(self._cleaned_output)
			pixmap = QPixmap.fromImage(qimg)
			
			# Set as the new base image
			self._image_view.set_pixmap(pixmap)
			
			# Update the original AI output to the cleaned result
			self._original_ai_output = self._cleaned_output.copy()
			
			# Clear the cleaned output since it's now the base
			self._cleaned_output = None
			
			self.statusBar().showMessage("Region cleanup applied to image.", 3000)
		else:
			QMessageBox.warning(self, "Apply Region Cleanup", "No original image available.")
	
	def _on_smooth_regions(self, method: str, strength: float, preserve_colors: bool) -> None:
		"""Handle region smoothing request."""
		# Check if we have a processed image available
		if self._simplified_output is None and not self._color_processing_applied:
			QMessageBox.warning(self, "Region Smoothing", "No processed image available. Complete color processing first.")
			return
		
		# Use the appropriate input image
		if self._simplified_output is not None:
			input_image = self._simplified_output
		else:
			input_image = self._original_ai_output
		
		if input_image is None:
			QMessageBox.warning(self, "Region Smoothing", "No image available for processing.")
			return
		
		try:
			# Perform region smoothing
			from processing.region_cleanup import smooth_region_boundaries
			
			smoothed_output = smooth_region_boundaries(
				input_image,
				method=method,
				strength=strength,
				preserve_colors=preserve_colors
			)
			
			# Convert to QImage for preview
			from utils.qt_image import numpy_rgba_to_qimage
			qimg = numpy_rgba_to_qimage(smoothed_output)
			
			# Set as preview
			self._image_view.set_preview_image(qimg)
			self._image_view.set_preview_enabled(True)
			
			# Update the cleaned output if it exists, otherwise update simplified output
			if self._cleaned_output is not None:
				self._cleaned_output = smoothed_output
			else:
				self._simplified_output = smoothed_output
			
			self.statusBar().showMessage(f"Region smoothing applied using {method} method.", 3000)
		
		except Exception as e:
			QMessageBox.critical(self, "Region Smoothing Error", f"An error occurred during region smoothing:\n{str(e)}")
	
	def _on_region_boundaries_toggled(self, show: bool) -> None:
		"""Handle region boundaries overlay toggle."""
		# This would show/hide region boundaries overlay
		# For now, just update the status
		if show:
			self.statusBar().showMessage("Region boundaries overlay enabled", 2000)
		else:
			self.statusBar().showMessage("Region boundaries overlay disabled", 2000)
	
	def _on_save_working_image(self) -> None:
		"""Handle save working image request."""
		# Determine which image to save
		image_to_save = None
		
		# Priority order: cleaned output > simplified output > original AI output
		if self._cleaned_output is not None:
			image_to_save = self._cleaned_output
			image_type = "cleaned"
		elif self._simplified_output is not None:
			image_to_save = self._simplified_output
			image_type = "simplified"
		elif self._original_ai_output is not None:
			image_to_save = self._original_ai_output
			image_type = "ai_processed"
		else:
			QMessageBox.warning(self, "Save Working Image", "No processed image available to save.")
			return
		
		# Show save dialog
		from PySide6.QtWidgets import QFileDialog
		file_path, _ = QFileDialog.getSaveFileName(
			self,
			"Save Working Image",
			f"working_image_{image_type}.tiff",
			"TIFF Files (*.tiff *.tif);;PNG Files (*.png);;BMP Files (*.bmp);;All Files (*)"
		)
		
		if file_path:
			try:
				# Try different approaches based on file format
				if file_path.lower().endswith('.bmp'):
					# BMP format - no transparency support
					from utils.qt_image import numpy_rgba_to_qimage
					qimg = numpy_rgba_to_qimage(image_to_save)
					qimg.save(file_path, "BMP")
				elif file_path.lower().endswith('.tiff') or file_path.lower().endswith('.tif'):
					# TIFF format - save directly with PIL to avoid Qt processing
					try:
						from PIL import Image
						# Save directly from numpy array to avoid any Qt processing
						pil_img = Image.fromarray(image_to_save, 'RGBA')
						pil_img.save(file_path, "TIFF")
					except ImportError:
						# Fallback to Qt TIFF saving
						from utils.qt_image import numpy_rgba_to_qimage
						qimg = numpy_rgba_to_qimage(image_to_save)
						qimg.save(file_path, "TIFF")
				else:
					# PNG format - save directly with PIL to avoid Qt processing
					try:
						from PIL import Image
						# Save directly from numpy array to avoid any Qt processing
						pil_img = Image.fromarray(image_to_save, 'RGBA')
						pil_img.save(file_path, "PNG", optimize=False, compress_level=0)
					except ImportError:
						# Fallback to Qt PNG saving
						from utils.qt_image import numpy_rgba_to_qimage
						qimg = numpy_rgba_to_qimage(image_to_save)
						qimg.save(file_path, "PNG")
				
				self.statusBar().showMessage(f"Working image saved to: {file_path}", 5000)
				QMessageBox.information(self, "Save Successful", f"Working image saved successfully to:\n{file_path}")
				
			except Exception as e:
				QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")
