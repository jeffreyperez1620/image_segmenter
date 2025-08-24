from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QPoint
from PySide6.QtGui import QMouseEvent, QPainterPath, QPen, QPainter, QColor, QPixmap, QImage
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget, QVBoxLayout, QGraphicsPathItem


class ImageView(QWidget):
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
		self._mode: str = "include"  # include|exclude|erase|crop
		self._brush_size: int = 24
		self._painting: bool = False
		self._last_pos_scene: Optional[QPointF] = None
		self._crop_origin: Optional[QPointF] = None
		self._crop_rect: Optional[QRectF] = None
		self._cropping_active: bool = False

		# Masks: 0=unmarked, 1=include, 2=exclude
		self._user_mask: Optional[np.ndarray] = None
		self._undo_stack: List[np.ndarray] = []
		self._redo_stack: List[np.ndarray] = []

		# Preview overlay
		self._preview_enabled: bool = False
		self._preview_pixmap: Optional[QPixmap] = None
		self._opacity_threshold: int = 128

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
				if event.type() == event.Type.MouseButtonPress and event.button() == Qt.LeftButton:
					self._on_mouse_press(event)
					return True
				elif event.type() == event.Type.MouseMove and (self._painting or (self._mode == "crop" and self._cropping_active)):
					self._on_mouse_move(event)
					return True
				elif event.type() == event.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
					self._on_mouse_release(event)
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

	def keyPressEvent(self, event) -> None:  # type: ignore[override]
		# Allow ESC to cancel current crop rectangle
		if event.key() == Qt.Key_Escape and self._mode == "crop":
			self._crop_rect = None
			self._crop_origin = None
			self._cropping_active = False
			self._crop_overlay_item.setVisible(False)
			self._update_overlay()
			return
		super().keyPressEvent(event)

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
		# If preview is enabled, show ONLY the preview
		if self._preview_enabled and self._preview_pixmap is not None:
			self._crop_overlay_item.setVisible(False)
			self._pix_item.setPixmap(self._preview_pixmap)
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

		# Compose masks/preview if any
		base = self._base_pixmap
		needs_compose = (self._user_mask is not None and (self._user_mask.any())) or (self._preview_enabled and self._preview_pixmap is not None)
		if not needs_compose:
			self._pix_item.setPixmap(base)
			self._view.viewport().update()
			return

		composed = QPixmap(base.size())
		composed.fill(QColor(0, 0, 0, 0))
		p = QPainter(composed)
		p.setRenderHint(QPainter.Antialiasing, True)
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
		# Preview branch is short-circuited above; no preview drawing here
		p.end()
		self._pix_item.setPixmap(composed)
		self._view.viewport().update()

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
			if event.angleDelta().y() > 0:
				self.scale(zoom_in_factor, zoom_in_factor)
			else:
				self.scale(zoom_out_factor, zoom_out_factor)
			return
		super().wheelEvent(event)
