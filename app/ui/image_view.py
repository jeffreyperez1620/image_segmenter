from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QPoint, Signal
from PySide6.QtGui import QMouseEvent, QPainterPath, QPen, QPainter, QColor, QPixmap, QImage, QCursor
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget, QVBoxLayout, QGraphicsPathItem


class ImageView(QWidget):
	colorPicked = Signal(QColor)  # Signal emitted when a color is picked with eyedropper
	eyedropperCancelled = Signal()  # Signal emitted when eyedropper mode is cancelled
	colorPreview = Signal(QColor)  # Signal emitted when hovering over a color in eyedropper mode
	floodFillRequested = Signal(int, int)  # Signal emitted when flood fill is requested at position (x, y)
	
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self._scene = QGraphicsScene(self)
		self._view = _GraphicsView(self._scene, self)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._view)

		self._pix_item: Optional[QGraphicsPixmapItem] = None
		self._base_pixmap: Optional[QPixmap] = None
		self._orig_qimage: Optional[QImage] = None

		# Interaction state
		self._mode: str = "include"  # include|exclude|erase|crop|eyedropper
		self._brush_size: int = 24
		self._painting: bool = False
		self._last_pos_scene: Optional[QPointF] = None
		self._crop_origin: Optional[QPointF] = None
		self._crop_rect: Optional[QRectF] = None
		self._cropping_active: bool = False
		
		# Panning state
		self._panning: bool = False
		self._last_pan_pos: Optional[QPointF] = None

		# Masks: 0=unmarked, 1=include, 2=exclude
		self._user_mask: Optional[np.ndarray] = None
		self._undo_stack: List[np.ndarray] = []
		self._redo_stack: List[np.ndarray] = []

		# Preview overlay
		self._preview_enabled: bool = False
		self._preview_pixmap: Optional[QPixmap] = None
		self._opacity_threshold: int = 128
		
		# Region boundaries overlay
		self._show_region_boundaries: bool = False
		self._region_boundaries_data: Optional[np.ndarray] = None

		# Crop overlay as separate graphics item
		self._crop_overlay_item = QGraphicsPathItem()
		self._crop_overlay_item.setZValue(10)
		self._crop_overlay_item.setBrush(QColor(255, 105, 180, 90))
		self._crop_overlay_item.setPen(Qt.NoPen)
		self._scene.addItem(self._crop_overlay_item)
		self._crop_overlay_item.setVisible(False)

		self._view.viewport().installEventFilter(self)

	def set_pixmap(self, pixmap: QPixmap) -> None:
		if self._pix_item is None:
			self._pix_item = QGraphicsPixmapItem(pixmap)
			self._scene.addItem(self._pix_item)
			# Ensure overlay item is in scene above the image
			if self._crop_overlay_item.scene() is None:
				self._scene.addItem(self._crop_overlay_item)
		else:
			self._pix_item.setPixmap(pixmap)
		# Reset all state for a fresh start
		self._base_pixmap = pixmap
		self._orig_qimage = pixmap.toImage()
		self._crop_rect = None
		self._crop_origin = None
		self._cropping_active = False
		self._preview_pixmap = None
		self._preview_enabled = False
		self._crop_overlay_item.setVisible(False)
		self._crop_overlay_item.setPath(QPainterPath())
		self._init_masks()
		self._view.fit_in_view()
		self._update_overlay()

	def set_mode(self, mode: str) -> None:
		self._mode = mode
		if mode == "crop":
			self._view.setDragMode(QGraphicsView.NoDrag)
			self._view.viewport().setCursor(Qt.CrossCursor)
			self._view.setFocus()
		elif mode == "eyedropper":
			self._view.setDragMode(QGraphicsView.NoDrag)
			# Set initial eyedropper cursor (will be updated with hovered color)
			initial_cursor = self._create_eyedropper_cursor(QColor(128, 128, 128))  # Gray default
			self._view.viewport().setCursor(initial_cursor)
			self._view.setFocus()
		elif mode == "flood_fill":
			self._view.setDragMode(QGraphicsView.NoDrag)
			self._view.viewport().setCursor(Qt.CrossCursor)
			self._view.setFocus()
		elif mode == "none":
			# No mode selected - allow normal scrolling
			self._view.setDragMode(QGraphicsView.ScrollHandDrag)
			self._view.viewport().unsetCursor()
		else:
			self._view.setDragMode(QGraphicsView.ScrollHandDrag)
			self._view.viewport().unsetCursor()
		self._update_overlay()

	def set_brush_size(self, size: int) -> None:
		self._brush_size = max(1, int(size))

	def set_preview_enabled(self, enabled: bool) -> None:
		self._preview_enabled = enabled
		self._update_overlay()

	def set_preview_image(self, qimage: Optional[QImage]) -> None:
		if qimage is not None:
			# Process the image to make pixels above threshold fully visible
			processed_qimage = self._process_alpha_for_preview(qimage)
			self._preview_pixmap = QPixmap.fromImage(processed_qimage)
		else:
			self._preview_pixmap = None
		self._update_overlay()

	def set_opacity_threshold(self, threshold: int) -> None:
		"""Set the opacity threshold for preview processing (0-255)."""
		self._opacity_threshold = max(0, min(255, threshold))
		# Re-process current preview if available
		if self._preview_pixmap is not None:
			# We need to get the original image back to re-process
			# This will be handled by the main window when threshold changes
			pass
	
	def set_region_boundaries_enabled(self, enabled: bool) -> None:
		"""Enable or disable region boundaries overlay."""
		self._show_region_boundaries = enabled
		self._update_overlay()
	
	def set_region_boundaries_data(self, rgba_data: Optional[np.ndarray]) -> None:
		"""Set the region boundaries data for overlay display."""
		self._region_boundaries_data = rgba_data
		if self._show_region_boundaries:
			self._update_overlay()

	def _process_alpha_for_preview(self, qimage: QImage) -> QImage:
		"""Process image to make pixels above threshold fully visible for preview."""
		# Convert to RGBA format if needed
		if qimage.format() != QImage.Format.Format_RGBA8888:
			qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
		
		# Get image data
		w = qimage.width()
		h = qimage.height()
		ptr = qimage.constBits()
		arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
		
		# Create a copy to modify
		processed_arr = arr.copy()
		
		# Make pixels above threshold fully opaque (alpha = 255)
		above_threshold = processed_arr[:, :, 3] > self._opacity_threshold
		processed_arr[above_threshold, 3] = 255
		
		# Make pixels at or below threshold fully transparent (alpha = 0)
		at_or_below_threshold = processed_arr[:, :, 3] <= self._opacity_threshold
		processed_arr[at_or_below_threshold, 3] = 0
		
		# Convert back to QImage
		processed_qimage = QImage(processed_arr.data, w, h, QImage.Format.Format_RGBA8888)
		return processed_qimage.copy()  # Deep copy to detach from numpy buffer

	def get_user_mask(self) -> Optional[np.ndarray]:
		return None if self._user_mask is None else self._user_mask.copy()

	def get_crop_rect_xywh(self) -> Optional[Tuple[int, int, int, int]]:
		if self._crop_rect is None:
			return None
		r = self._crop_rect.normalized()
		return (int(r.x()), int(r.y()), int(r.width()), int(r.height()))

	def clear_marks(self) -> None:
		if self._user_mask is not None:
			self._push_undo()
			self._user_mask[:] = 0
			self._redo_stack.clear()
			self._update_overlay()

	def undo(self) -> None:
		if not self._undo_stack:
			return
		if self._user_mask is None:
			return
		self._redo_stack.append(self._user_mask.copy())
		self._user_mask = self._undo_stack.pop()
		self._update_overlay()

	def redo(self) -> None:
		if not self._redo_stack or self._user_mask is None:
			return
		self._undo_stack.append(self._user_mask.copy())
		self._user_mask = self._redo_stack.pop()
		self._update_overlay()

	def apply_crop(self) -> None:
		if self._crop_rect is None or self._orig_qimage is None or self._pix_item is None:
			return
		r = self._crop_rect.normalized().toRect()
		if r.isEmpty():
			return
		cropped = self._orig_qimage.copy(r)
		self.set_pixmap(QPixmap.fromImage(cropped))
		self._crop_rect = None
		self._cropping_active = False
		self._crop_overlay_item.setVisible(False)

	def eventFilter(self, obj, event):  # type: ignore[override]
		if obj is self._view.viewport() and self._orig_qimage is not None:
			if isinstance(event, QMouseEvent):
				if event.type() == event.Type.MouseButtonPress:
					if event.button() == Qt.LeftButton:
						self._on_mouse_press(event)
						return True
					elif event.button() == Qt.RightButton:
						self._on_right_mouse_press(event)
						return True
				elif event.type() == event.Type.MouseMove:
					if (self._painting or (self._mode == "crop" and self._cropping_active) or self._mode == "eyedropper"):
						self._on_mouse_move(event)
						return True
					elif self._panning:
						self._on_right_mouse_move(event)
						return True
				elif event.type() == event.Type.MouseButtonRelease:
					if event.button() == Qt.LeftButton:
						self._on_mouse_release(event)
						return True
					elif event.button() == Qt.RightButton:
						self._on_right_mouse_release(event)
						return True
		return super().eventFilter(obj, event)

	def _on_mouse_press(self, event: QMouseEvent) -> None:
		scene_pos = self._view.mapToScene(event.position().toPoint())
		if self._mode == "crop":
			self._crop_origin = scene_pos
			self._crop_rect = QRectF(scene_pos, scene_pos)
			self._cropping_active = True
			self._update_overlay()
			return
		elif self._mode == "eyedropper":
			self._pick_color_at_position(scene_pos)
			return
		elif self._mode == "flood_fill":
			# Emit flood fill request with the clicked position
			x = int(scene_pos.x())
			y = int(scene_pos.y())
			self.floodFillRequested.emit(x, y)
			return
		# painting
		self._painting = True
		self._last_pos_scene = scene_pos
		self._push_undo()
		self._apply_brush(scene_pos)

	def _on_mouse_move(self, event: QMouseEvent) -> None:
		scene_pos = self._view.mapToScene(event.position().toPoint())
		if self._mode == "crop" and self._crop_origin is not None:
			self._crop_rect = QRectF(self._crop_origin, scene_pos)
			self._update_overlay()
			return
		elif self._mode == "eyedropper":
			self._preview_color_at_position(scene_pos)
			return
		self._apply_brush(scene_pos)
		self._last_pos_scene = scene_pos

	def _on_mouse_release(self, event: QMouseEvent) -> None:
		if self._mode == "crop":
			self._cropping_active = False
			self._crop_origin = None
			self._update_overlay()
			return
		self._painting = False
		self._last_pos_scene = None

	def _on_right_mouse_press(self, event: QMouseEvent) -> None:
		"""Handle right mouse button press for panning."""
		self._panning = True
		self._last_pan_pos = event.position()  # event.position() already returns QPointF
		self._view.setDragMode(QGraphicsView.NoDrag)  # Disable default drag mode
		self._view.viewport().setCursor(Qt.ClosedHandCursor)

	def _on_right_mouse_move(self, event: QMouseEvent) -> None:
		"""Handle right mouse button move for panning."""
		if not self._panning or self._last_pan_pos is None:
			return
		
		current_pos = event.position()  # event.position() already returns QPointF
		delta = current_pos - self._last_pan_pos
		
		# Pan the view
		scroll_bar_h = self._view.horizontalScrollBar()
		scroll_bar_v = self._view.verticalScrollBar()
		scroll_bar_h.setValue(scroll_bar_h.value() - int(delta.x()))
		scroll_bar_v.setValue(scroll_bar_v.value() - int(delta.y()))
		
		self._last_pan_pos = current_pos

	def _on_right_mouse_release(self, event: QMouseEvent) -> None:
		"""Handle right mouse button release for panning."""
		self._panning = False
		self._last_pan_pos = None
		self._view.viewport().unsetCursor()
		# Restore appropriate drag mode based on current mode
		if self._mode == "crop" or self._mode == "eyedropper" or self._mode == "flood_fill":
			self._view.setDragMode(QGraphicsView.NoDrag)
		else:
			self._view.setDragMode(QGraphicsView.ScrollHandDrag)

	def keyPressEvent(self, event) -> None:  # type: ignore[override]
		# Allow ESC to cancel current crop rectangle
		if event.key() == Qt.Key_Escape and self._mode == "crop":
			self._crop_rect = None
			self._crop_origin = None
			self._cropping_active = False
			self._crop_overlay_item.setVisible(False)
			self._update_overlay()
			return
		# Allow ESC to cancel eyedropper mode
		elif event.key() == Qt.Key_Escape and self._mode == "eyedropper":
			self.set_mode("include")
			self.eyedropperCancelled.emit()
			return
		# Allow ESC to cancel flood fill mode
		elif event.key() == Qt.Key_Escape and self._mode == "flood_fill":
			self.set_mode("include")
			return
		super().keyPressEvent(event)

	def _pick_color_at_position(self, scene_pos: QPointF) -> None:
		"""Pick color from the image at the given scene position."""
		# Determine which image to pick from based on current display
		if self._preview_enabled and self._preview_pixmap is not None:
			# Pick from the preview image
			source_image = self._preview_pixmap.toImage()
		else:
			# Pick from the original image
			source_image = self._orig_qimage
		
		if source_image is None:
			return
		
		# Convert scene position to image coordinates
		x = int(scene_pos.x())
		y = int(scene_pos.y())
		
		# Check bounds
		if x < 0 or y < 0 or x >= source_image.width() or y >= source_image.height():
			return
		
		# Get the color at the pixel using pixel() method for more reliable color extraction
		pixel_value = source_image.pixel(x, y)
		
		# Extract RGB values from the pixel value
		red = (pixel_value >> 16) & 0xFF
		green = (pixel_value >> 8) & 0xFF
		blue = pixel_value & 0xFF
		alpha = (pixel_value >> 24) & 0xFF
		
		# Create QColor from extracted values
		color = QColor(red, green, blue, alpha)
		
		
		# Emit the color picked signal
		self.colorPicked.emit(color)

	def _preview_color_at_position(self, scene_pos: QPointF) -> None:
		"""Preview color from the image at the given scene position without picking it."""
		# Always use the same source image as the pick function for consistency
		if self._preview_enabled and self._preview_pixmap is not None:
			# Preview from the preview image
			source_image = self._preview_pixmap.toImage()
		else:
			# Preview from the original image
			source_image = self._orig_qimage
		
		if source_image is None:
			return
		
		# Convert scene position to image coordinates
		x = int(scene_pos.x())
		y = int(scene_pos.y())
		
		# Check bounds
		if x < 0 or y < 0 or x >= source_image.width() or y >= source_image.height():
			return
		
		# Get the color at the pixel using pixel() method for more reliable color extraction
		pixel_value = source_image.pixel(x, y)
		
		# Extract RGB values from the pixel value
		# QImage.pixel() returns a QRgb value (32-bit integer with ARGB format)
		red = (pixel_value >> 16) & 0xFF
		green = (pixel_value >> 8) & 0xFF
		blue = pixel_value & 0xFF
		alpha = (pixel_value >> 24) & 0xFF
		
		# Create QColor from extracted values
		color = QColor(red, green, blue, alpha)
		
		
		# Update the eyedropper cursor color
		self._update_eyedropper_cursor(color)
		
		# Emit the color preview signal
		self.colorPreview.emit(color)
	
	def _create_eyedropper_cursor(self, color: QColor) -> QCursor:
		"""Create a custom eyedropper cursor with the specified color."""
		# Create a 40x40 pixmap for the cursor (larger for better visibility)
		cursor_pixmap = QPixmap(40, 40)
		cursor_pixmap.fill(Qt.transparent)
		
		painter = QPainter(cursor_pixmap)
		painter.setRenderHint(QPainter.Antialiasing)
		
		# Draw the eyedropper shape with the exact color
		# Main body (rectangle) - use the exact color being picked
		painter.setBrush(color)  # Use the exact hovered color
		painter.setPen(QPen(Qt.black, 2))
		painter.drawRect(8, 8, 20, 16)
		
		# Tip (large circle) - this shows the exact color being picked
		painter.setBrush(color)  # Use the exact hovered color
		painter.setPen(QPen(Qt.black, 2))
		painter.drawEllipse(12, 24, 12, 12)
		
		# Handle (small rectangle at top) - also use exact color
		painter.setBrush(color)  # Use the exact hovered color
		painter.setPen(QPen(Qt.black, 2))
		painter.drawRect(10, 6, 16, 4)
		
		# Add a small white dot in the center of the tip to show the exact pixel
		painter.setBrush(Qt.white)
		painter.setPen(Qt.white)
		painter.drawEllipse(17, 29, 2, 2)
		
		painter.end()
		
		# Create cursor with hotspot at the center of the tip
		return QCursor(cursor_pixmap, 18, 30)
	
	def _update_eyedropper_cursor(self, color: QColor) -> None:
		"""Update the eyedropper cursor to show the current hovered color."""
		if self._mode == "eyedropper":
			cursor = self._create_eyedropper_cursor(color)
			self._view.viewport().setCursor(cursor)

	def _apply_brush(self, scene_pos: QPointF) -> None:
		if self._user_mask is None:
			return
		x = int(scene_pos.x())
		y = int(scene_pos.y())
		h, w = self._user_mask.shape
		radius = max(1, self._brush_size // 2)
		# Determine value based on mode
		if self._mode == "include":
			paint_value = 1
		elif self._mode == "exclude":
			paint_value = 2
		elif self._mode == "erase":
			paint_value = 0
		else:
			return
		# Interpolate along stroke to avoid gaps
		if self._last_pos_scene is not None:
			lx = int(self._last_pos_scene.x())
			ly = int(self._last_pos_scene.y())
			dx = x - lx
			dy = y - ly
			dist = float(np.hypot(dx, dy))
			step_px = max(1, radius // 2)
			steps = max(1, int(dist / step_px))
			for i in range(steps + 1):
				t = i / float(steps)
				ix = int(round(lx + t * dx))
				iy = int(round(ly + t * dy))
				self._paint_disk(ix, iy, radius, paint_value)
		else:
			self._paint_disk(x, y, radius, paint_value)
		self._update_overlay()

	def _paint_disk(self, cx: int, cy: int, radius: int, value: int) -> None:
		if self._user_mask is None:
			return
		h, w = self._user_mask.shape
		x0 = max(0, cx - radius)
		y0 = max(0, cy - radius)
		x1 = min(w, cx + radius + 1)
		y1 = min(h, cy + radius + 1)
		if x0 >= x1 or y0 >= y1:
			return
		yy, xx = np.ogrid[y0:y1, x0:x1]
		circle = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= radius * radius
		sub = self._user_mask[y0:y1, x0:x1]
		if value == 0:
			sub[circle] = 0
		else:
			sub[circle] = value

	def _push_undo(self) -> None:
		if self._user_mask is not None:
			self._undo_stack.append(self._user_mask.copy())
			# Limit history size
			if len(self._undo_stack) > 50:
				self._undo_stack.pop(0)
			self._redo_stack.clear()

	def _init_masks(self) -> None:
		if self._orig_qimage is None:
			return
		w = self._orig_qimage.width()
		h = self._orig_qimage.height()
		self._user_mask = np.zeros((h, w), dtype=np.uint8)
		self._undo_stack.clear()
		self._redo_stack.clear()

	def _update_overlay(self) -> None:
		if self._pix_item is None or self._base_pixmap is None:
			return
		# If preview is enabled, show the preview with any overlays
		if self._preview_enabled and self._preview_pixmap is not None:
			self._crop_overlay_item.setVisible(False)
			
			# Check if we need to compose overlays on top of preview
			needs_compose = (self._user_mask is not None and (self._user_mask.any())) or (self._show_region_boundaries and self._region_boundaries_data is not None)
			
			if not needs_compose:
				# No overlays needed, just show preview
				self._pix_item.setPixmap(self._preview_pixmap)
				self._view.viewport().update()
				return
			else:
				# Compose overlays on top of preview
				composed = QPixmap(self._preview_pixmap.size())
				composed.fill(QColor(0, 0, 0, 0))
				p = QPainter(composed)
				# Disable anti-aliasing to preserve sharp alpha edges
				p.setRenderHint(QPainter.Antialiasing, False)
				p.drawPixmap(0, 0, self._preview_pixmap)
				
				# Draw user mask overlays if present
				if self._user_mask is not None:
					overlay = QPixmap(self._preview_pixmap.size())
					overlay.fill(QColor(0, 0, 0, 0))
					po = QPainter(overlay)
					po.setRenderHint(QPainter.Antialiasing, False)
					self._draw_mask_color(po, self._user_mask, value=1, color=QColor(0, 255, 0, 80))
					self._draw_mask_color(po, self._user_mask, value=2, color=QColor(255, 0, 0, 80))
					po.end()
					p.drawPixmap(0, 0, overlay)
				
				# Draw region boundaries if enabled
				if self._show_region_boundaries and self._region_boundaries_data is not None:
					self._draw_region_boundaries(p)
				
				p.end()
				self._pix_item.setPixmap(composed)
				self._view.viewport().update()
				return
		# Handle crop overlay path
		if self._crop_rect is not None:
			outer = QRectF(0, 0, self._base_pixmap.width(), self._base_pixmap.height())
			r = self._crop_rect.normalized()
			path = QPainterPath()
			path.setFillRule(Qt.OddEvenFill)
			path.addRect(outer)
			path.addRect(r)
			self._crop_overlay_item.setPath(path)
			self._crop_overlay_item.setVisible(True)
			# During active crop drag, avoid recomposition for performance
			if self._mode == "crop" and self._cropping_active:
				self._pix_item.setPixmap(self._base_pixmap)
				self._view.viewport().update()
				return
		else:
			self._crop_overlay_item.setVisible(False)

		# Compose masks/preview/region boundaries if any
		base = self._base_pixmap
		needs_compose = (self._user_mask is not None and (self._user_mask.any())) or (self._preview_enabled and self._preview_pixmap is not None) or (self._show_region_boundaries and self._region_boundaries_data is not None)
		if not needs_compose:
			self._pix_item.setPixmap(base)
			self._view.viewport().update()
			return

		composed = QPixmap(base.size())
		composed.fill(QColor(0, 0, 0, 0))
		p = QPainter(composed)
		# Disable anti-aliasing to preserve sharp alpha edges
		p.setRenderHint(QPainter.Antialiasing, False)
		p.drawPixmap(0, 0, base)
		if self._user_mask is not None:
			overlay = QPixmap(base.size())
			overlay.fill(QColor(0, 0, 0, 0))
			po = QPainter(overlay)
			po.setRenderHint(QPainter.Antialiasing, False)
			self._draw_mask_color(po, self._user_mask, value=1, color=QColor(0, 255, 0, 80))
			self._draw_mask_color(po, self._user_mask, value=2, color=QColor(255, 0, 0, 80))
			po.end()
			p.drawPixmap(0, 0, overlay)
		
		# Draw region boundaries if enabled
		if self._show_region_boundaries and self._region_boundaries_data is not None:
			self._draw_region_boundaries(p)
		
		# Draw preview if enabled
		if self._preview_enabled and self._preview_pixmap is not None:
			p.drawPixmap(0, 0, self._preview_pixmap)
		
		p.end()
		self._pix_item.setPixmap(composed)
		self._view.viewport().update()

	def _draw_region_boundaries(self, painter: QPainter) -> None:
		"""Draw region boundaries overlay."""
		if self._region_boundaries_data is None:
			return
		
		try:
			# Convert numpy array to QImage for drawing
			from utils.qt_image import numpy_rgba_to_qimage
			boundaries_qimage = numpy_rgba_to_qimage(self._region_boundaries_data)
			boundaries_pixmap = QPixmap.fromImage(boundaries_qimage)
			
			# Draw the boundaries with some transparency
			painter.setOpacity(0.7)
			painter.drawPixmap(0, 0, boundaries_pixmap)
			painter.setOpacity(1.0)
		except Exception as e:
			print(f"Error drawing region boundaries: {e}")
			# Continue without boundaries if there's an error
	
	@staticmethod
	def _draw_mask_color(p: QPainter, mask: np.ndarray, value: int, color: QColor) -> None:
		if mask is None:
			return
		# Simple block drawing for marked pixels (optimize later with QImage or textures)
		h, w = mask.shape
		br = 1
		p.setPen(Qt.NoPen)
		p.setBrush(color)
		for y in range(h):
			row = mask[y]
			x = 0
			while x < w:
				if row[x] == value:
					x2 = x + 1
					while x2 < w and row[x2] == value:
						x2 += 1
					p.drawRect(x, y, x2 - x, 1)
					x = x2
				else:
					x += 1


class _GraphicsView(QGraphicsView):
	def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None) -> None:
		super().__init__(scene, parent)
		self.setRenderHints(self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
		self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
		self.setDragMode(QGraphicsView.ScrollHandDrag)
		self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
		self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

	def fit_in_view(self) -> None:
		if self.scene() is None or self.scene().itemsBoundingRect().isEmpty():
			return
		self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

	def wheelEvent(self, event) -> None:  # type: ignore[override]
		zoom_in_factor = 1.25
		zoom_out_factor = 1.0 / zoom_in_factor
		if event.modifiers() & Qt.ControlModifier:
			# Always zoom to mouse position when Ctrl is held
			self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
			self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
			
			if event.angleDelta().y() > 0:
				self.scale(zoom_in_factor, zoom_in_factor)
			else:
				self.scale(zoom_out_factor, zoom_out_factor)
			return
		super().wheelEvent(event)
