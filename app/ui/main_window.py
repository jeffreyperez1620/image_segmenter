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
from ui.arrange_regions.arrange_regions_panel import ArrangeRegionsPanel
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
		
		# Set minimum window size
		self.setMinimumSize(800, 600)
		
		# Get screen geometry
		screen = QApplication.primaryScreen()
		if screen is not None:
			screen_geometry = screen.availableGeometry()
			screen_width = screen_geometry.width()
			screen_height = screen_geometry.height()
			
			# Calculate initial size: half of screen dimensions
			initial_width = screen_width // 4 * 3
			initial_height = screen_height // 4 * 3
			
			# Ensure minimum size
			initial_width = max(initial_width, 800)
			initial_height = max(initial_height, 800)
			
			# Calculate position to center the window
			x = (screen_width - initial_width) // 2
			y = (screen_height - initial_height) // 2
			
			# Set geometry (position and size)
			self.setGeometry(x, y, initial_width, initial_height)
		else:
			# Fallback if screen is not available
			self.resize(1200, 800)
		
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
		
		# Arrange regions tab
		self._arrange_regions_panel = ArrangeRegionsPanel(self, self._app_state, self._image_view)
		self._tab_widget.addTab(self._arrange_regions_panel, "4. Arrange Regions")
		
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
		
		# Check if the target tab can be entered
		targetTab = self._tab_widget.widget(new_index)
		if targetTab is not None and isinstance(targetTab, BaseStep):
			if not targetTab.validate_entry():
				return False  # Prevent the tab change
		
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
				# Disconnect status bar message signal
				if hasattr(prevTab, 'statusBarMessage'):
					try:
						prevTab.statusBarMessage.disconnect(self._on_status_bar_message)
					except:
						pass
				prevTab._on_close()
		currentTab = self._tab_widget.widget(index)
		if currentTab is not None and isinstance(currentTab, BaseStep):
			# Connect status bar message signal
			if hasattr(currentTab, 'statusBarMessage'):
				currentTab.statusBarMessage.connect(self._on_status_bar_message)
			currentTab._on_open()		
		self._previous_tab_index = index
		if index == 0:  # Background Removal tab
			self.statusBar().showMessage("Step 1: Remove background using AI or manual tools", 3000)
		elif index == 1:  # Color Processing tab
				self.statusBar().showMessage("Step 2: Process colors using algorithms or create custom palette", 3000)
		elif index == 2:  # Region Cleanup tab
				self.statusBar().showMessage("Step 3: Clean up small regions for laser engraving", 3000)
		elif index == 3:  # Arrange Regions tab
				self.statusBar().showMessage("Step 4: Arrange regions by color for laser engraving layout", 3000)

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

	
	def _on_save_working_image(self) -> None:
		"""Handle save working image request."""
		# Determine which image to save
		image_to_save = None
		
		# Priority order: cleaned output > simplified output > original AI output
		image_to_save = self._app_state.working_image
		if image_to_save is None:
			QMessageBox.warning(self, "Save Working Image", "No image available to save.")
			return
		
		# Show save dialog
		from PySide6.QtWidgets import QFileDialog
		file_path, _ = QFileDialog.getSaveFileName(
			self,
			"Save Working Image",
			f"working_image.tiff",
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
		# Update arrange regions overlay view
		self._arrange_regions_panel._region_overlay_view.setGeometry(0, 0, self._image_view.width(), self._image_view.height())
		# Call the original resize event
		QWidget.resizeEvent(self._image_view, event)
