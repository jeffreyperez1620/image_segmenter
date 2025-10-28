from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QImage, QColor
from PySide6.QtWidgets import (
	QDockWidget,
	QFileDialog,
	QMainWindow,
	QMessageBox,
	QToolBar,
	QApplication,
	QTabWidget,
	QTabBar,
	QDialog,
	QDialogButtonBox,
	QGridLayout,
	QLabel,
	QPushButton,
	QScrollArea,
	QWidget,
	QVBoxLayout,
)

from ui.image_view import ImageView
from ui.background_removal.bg_tools_panel import BgToolsPanel
from ui.color_processing.color_processing_panel import ColorProcessingPanel
from ui.region_cleanup.region_cleanup_panel import RegionCleanupPanel
from ui.progress_dialog import ProgressDialog
from ui.validating_tab_widget import ValidatingTabWidget
from ui.base_step import BaseStep
from model import AppState
from utils.qt_image import qimage_to_numpy_bgr, composite_foreground_over_transparent, numpy_rgba_to_qimage
from processing.color_simplify import simplify_colors_adaptive, get_color_statistics
from processing.region_cleanup import analyze_regions, merge_small_regions, flood_fill_region, get_region_boundaries


class MainWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Image Segmenter & SVG Layout")
		self.resize(1200, 400)
		
		# Initialize application state
		self._app_state = AppState()

		# Create a container widget for the image view and b`ackground removal overlay
		image_container = QWidget(self)
		image_layout = QVBoxLayout(image_container)
		image_layout.setContentsMargins(0, 0, 0, 0)
		
		self._image_view = ImageView(self._app_state, self)
		image_layout.addWidget(self._image_view)
		
		# Connect resize events to keep the overlay properly sized
		self._image_view.resizeEvent = self._on_image_view_resize
		
		self.setCentralWidget(image_container)

		self._create_actions()
		self._create_menus_and_toolbar()
		self._create_docks()

		# Initialize the first tab
		self._on_tab_changed(0)

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
		self._tab_widget = ValidatingTabWidget(self)
		self._tab_widget.setObjectName("ToolsTabWidget")
		self._tab_widget.setTabPosition(QTabWidget.North)
		
		# Background removal tab
		self._bg_panel = BgToolsPanel(self, self._app_state, self._image_view)
		self._tab_widget.addTab(self._bg_panel, "1. Background Removal")
		
		# Color processing tab (combines simplification and custom palette)
		self._color_processing_panel = ColorProcessingPanel(self, self._app_state, self._image_view)
		self._tab_widget.addTab(self._color_processing_panel, "2. Color Processing")
		
		# Region cleanup tab
		self._region_cleanup_panel = RegionCleanupPanel(self, self._app_state, self._image_view)
		self._tab_widget.addTab(self._region_cleanup_panel, "3. Region Cleanup")
		
		# Create dock widget for the tabbed interface
		self._dock_tools = QDockWidget("Workflow Tools", self)
		self._dock_tools.setObjectName("DockTools")
		self._dock_tools.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
		self._dock_tools.setWidget(self._tab_widget)
		# Set maximum height to prevent the dock from being too tall
		self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_tools)
		
		
		# Connect tab changes to update workflow guidance
		self._tab_widget.currentChanged.connect(self._on_tab_changed)
		# Set up validation callback for tab changes
		self._tab_widget.set_validation_callback(self._validate_tab_change)

	def _validate_tab_change(self, current_index: int, new_index: int) -> bool:
		"""Validate tab change and return True if allowed, False if cancelled."""
		
		# Check for unapplied changes when trying to switch tabs
		currentTab = self._tab_widget.widget(current_index)
		if currentTab is not None and isinstance(currentTab, BaseStep) and currentTab.has_unapplied_changes():
			choice = self._show_unapplied_changes_dialog()
			
			if choice == "cancel":
				return False  # Prevent the tab change
			elif choice == "discard":
				# User chose to discard changes, reset working image to base
				self._app_state.reset_working_image()
			elif choice == "apply":
				# User chose to apply changes, update base image
				self._app_state.apply_working_to_base()
			currentTab.mark_changes_applied()
		
		return True  # Allow the tab change

	def _show_unapplied_changes_dialog(self) -> str:
		"""Show dialog with three choices for unapplied changes."""
		from ui.unapplied_changes_dialog import UnappliedChangesDialog
		return UnappliedChangesDialog.show_dialog(self)

	def _on_tab_change_requested(self, index: int) -> None:
		"""Handle tab change requests to validate before switching."""
		# Check for unapplied changes when trying to switch tabs
		if BaseStep._shared_has_unapplied_changes:
			choice = self._show_unapplied_changes_dialog()
			
			if choice == "cancel":
				# Cancel the tab change
				self._tab_widget.allowTabChange(False)
				return
			elif choice == "apply":
				# User chose to apply changes, update base image
				self._app_state.apply_working_to_base()
			else:  # choice == "discard"
				# User chose to discard, reset working image to base
				self._app_state.reset_working_image()
		# Allow the tab change to proceed
		self._tab_widget.allowTabChange(True)

	def _on_tab_changed(self, index: int) -> None:
		if hasattr(self, '_previous_tab_index'):
			prevTab = self._tab_widget.widget(self._previous_tab_index)
			if prevTab is not None and isinstance(prevTab, BaseStep):
				prevTab._on_close()
		currentTab = self._tab_widget.widget(index)
		if currentTab is not None and isinstance(currentTab, BaseStep):
			currentTab._on_open()		
		self._previous_tab_index = index
		if index == 0:  # Background Removal tab
			self.statusBar().showMessage("Step 1: Remove background using AI or manual tools", 3000)
		elif index == 1:  # Color Processing tab
				self.statusBar().showMessage("Step 2: Process colors using algorithms or create custom palette", 3000)
		elif index == 2:  # Region Cleanup tab
				self.statusBar().showMessage("Step 3: Clean up small regions for laser engraving", 3000)

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

		# Initialize base/working image system using AppState
		from utils.qt_image import qimage_to_numpy_rgba
		rgba = qimage_to_numpy_rgba(image)
		self._app_state.base_image = rgba.copy()
		self._app_state.working_image = rgba.copy()

		# Initialize background removal view with image size
		self._bg_panel._background_removal_view.set_image_size(image.width(), image.height())
		
		self.statusBar().showMessage(f"Loaded {file_path}", 5000)

	def _on_about(self) -> None:
		QMessageBox.information(
			self,
			"About",
			"""
			Image Segmenter & SVG Layout
			
			A tool to simplify images, segment by color, arrange segments for minimal layout, and export SVG for laser engraving.
			""".strip(),
		)	

	def _on_status_bar_message(self, message: str) -> None:
		"""Handle status bar message."""
		self.statusBar().showMessage(message, 3000)

	def _on_tendril_cleanup(self) -> None:
		"""Handle tendril cleanup request."""
		# Check if we have a processed image available
		if self._simplified_output is None and not self._color_processing_applied:
			QMessageBox.warning(self, "Smoothing", "No processed image available. Complete color processing first.")
			return
		
		# Use the appropriate input image (same priority as _get_current_working_image)
		if self._cleaned_output is not None:
			# Use the cleaned output (smoothed) if available
			input_image = self._cleaned_output
		elif self._simplified_output is not None:
			# Use the simplified output (color processed) if available
			input_image = self._simplified_output
		else:
			# Fallback to working image
			working_image = BaseStep.get_shared_working_image()
			input_image = working_image
		
		if input_image is None:
			QMessageBox.warning(self, "Smoothing", "No image available for processing.")
			return
		
		# Get parameters from the panel
		threshold = self._region_cleanup_panel.get_tendril_threshold()
		max_iterations = self._region_cleanup_panel.get_tendril_max_iterations()
		
		# Show progress dialog
		self._progress_dialog = ProgressDialog("Smoothing", self)
		self._progress_dialog.show()
		
		# Create and start worker thread
		from ui.tendril_worker import TendrilWorker
		self._tendril_worker = TendrilWorker(input_image, threshold, max_iterations)
		self._tendril_worker.progress_updated.connect(self._on_tendril_progress_updated)
		self._tendril_worker.cleanup_completed.connect(self._on_tendril_cleanup_completed)
		self._tendril_worker.cleanup_failed.connect(self._on_tendril_cleanup_failed)
		self._tendril_worker.start()
	
	def _on_tendril_progress_updated(self, current: int, total: int, message: str) -> None:
		"""Handle tendril cleanup progress updates."""
		if hasattr(self, '_progress_dialog') and self._progress_dialog:
			self._progress_dialog.update_progress(current, total, message)
	
	def _on_tendril_cleanup_completed(self, cleaned_output: np.ndarray, iterations_used: int, status_message: str) -> None:
		"""Handle successful tendril cleanup completion."""
		# Close progress dialog
		if hasattr(self, '_progress_dialog') and self._progress_dialog:
			self._progress_dialog.close()
			self._progress_dialog = None
		
		# Update the cleaned output
		self._cleaned_output = cleaned_output
		
		# Convert to QImage for preview
		from utils.qt_image import numpy_rgba_to_qimage
		qimg = numpy_rgba_to_qimage(cleaned_output)
		
		# Set as preview
		self._image_view.set_preview_image(qimg)
		self._image_view.set_preview_enabled(True)
		
		# Update region boundaries if overlay is enabled
		if self._region_cleanup_panel.get_show_region_overlay():
			boundaries_data = self._generate_region_boundaries(cleaned_output)
			self._image_view.set_region_boundaries_data(boundaries_data)
		
		self.statusBar().showMessage(f"Smoothing completed in {iterations_used} iterations. Use 'Apply Cleanup' to make changes permanent.", 5000)
	
	def _on_tendril_cleanup_failed(self, error_message: str) -> None:
		"""Handle tendril cleanup failure."""
		# Close progress dialog
		if hasattr(self, '_progress_dialog') and self._progress_dialog:
			self._progress_dialog.close()
			self._progress_dialog = None
		
		QMessageBox.critical(self, "Smoothing Error", f"An error occurred during smoothing:\n{error_message}")
	
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
		else:
			working_image = BaseStep.get_shared_working_image()
			if working_image is not None:
				image_to_save = working_image
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
	
	def _on_image_view_resize(self, event):
		"""Handle image view resize events."""
		# Update the background removal view size to match
		self._bg_panel._background_removal_view.setGeometry(0, 0, self._image_view.width(), self._image_view.height())
		# Call the original resize event
		QWidget.resizeEvent(self._image_view, event)
