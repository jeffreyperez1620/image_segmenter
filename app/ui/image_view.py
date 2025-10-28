from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QPoint, Signal
from PySide6.QtGui import QMouseEvent, QPainterPath, QPen, QPainter, QColor, QPixmap, QImage, QCursor
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget, QVBoxLayout, QGraphicsPathItem

from model import AppState
from utils.qt_image import numpy_rgba_to_qimage


class ImageView(QWidget):
	floodFillRequested = Signal(int, int)  # Signal emitted when flood fill is requested at position (x, y)
	
	def __init__(self, app_state: AppState, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)
		self._app_state = app_state
		self._scene = QGraphicsScene(self)
		self._view = _GraphicsView(self._scene, self)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._view)

		self._pix_item = QGraphicsPixmapItem()
		self._scene.addItem(self._pix_item)
		self._base_pixmap: Optional[QPixmap] = None

		# Interaction state
		self._mode: str = "none"  # none|flood_fill
		
		# Panning state
		self._panning: bool = False
		self._last_pan_pos: Optional[QPointF] = None

		# Working overlay
		self._show_working: bool = True
		self._working_pixmap: Optional[QPixmap] = None

		self._view.viewport().installEventFilter(self)

		self._app_state.base_image_changed.connect(self._on_base_image_changed)
		self._app_state.working_image_changed.connect(self._on_working_image_changed)

	def set_pixmap(self, pixmap: QPixmap) -> None:
		self._pix_item.setPixmap(pixmap)
		# Reset all state for a fresh start
		self._base_pixmap = pixmap
		self._working_pixmap = None
		self._show_working = False
		self._view.fit_in_view()
		self._update_overlay()

	def switch_to_image(self, pixmap: QPixmap) -> None:
		"""Switch to a different image without resetting other state."""
		if self._pix_item is not None:
			self._pix_item.setPixmap(pixmap)
		self._base_pixmap = pixmap
		self._update_overlay()
	
	def is_panning(self) -> bool:
		return self._panning

	def set_mode(self, mode: str) -> None:
		self._mode = mode
		if mode == "flood_fill":
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

	def get_current_display_image(self) -> Optional[np.ndarray]:
		"""Get the image that should be displayed (base or working based on checkbox)."""
		if self._show_working:
			return self._app_state.working_image
		else:
			return self._app_state.base_image

	def set_show_working(self, enabled: bool) -> None:
		self._show_working = enabled
		self._update_overlay()

	def set_temporary_image(self, image) -> None:
		"""Set a temporary working image (numpy array or QImage)."""
		if image is not None:
			# Ensure we have a QImage first
			from utils.qt_image import ensure_qimage
			qimg = ensure_qimage(image)
			self._working_pixmap = QPixmap.fromImage(qimg)
		else:
			self._working_pixmap = None
		self._update_overlay()


	def eventFilter(self, obj, event):  # type: ignore[override]
		if obj is self._view.viewport() and self._base_pixmap is not None:
			if isinstance(event, QMouseEvent):
				if event.type() == event.Type.MouseButtonPress:
					if event.button() == Qt.LeftButton:
						self._on_mouse_press(event)
						return True
					elif event.button() == Qt.RightButton:
						self._on_right_mouse_press(event)
						return True
				elif event.type() == event.Type.MouseMove:
					if self._panning:
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
		if self._mode == "flood_fill":
			# Emit flood fill request with the clicked position
			x = int(scene_pos.x())
			y = int(scene_pos.y())
			self.floodFillRequested.emit(x, y)
			return

	def _on_mouse_move(self, event: QMouseEvent) -> None:
		scene_pos = self._view.mapToScene(event.position().toPoint())

	def _on_mouse_release(self, event: QMouseEvent) -> None:
		pass  # No specific action needed for remaining modes

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
		if self._mode == "flood_fill":
			self._view.setDragMode(QGraphicsView.NoDrag)
		else:
			self._view.setDragMode(QGraphicsView.ScrollHandDrag)

	def keyPressEvent(self, event) -> None:  # type: ignore[override]
		# Allow ESC to cancel flood fill mode
		if event.key() == Qt.Key_Escape and self._mode == "flood_fill":
			self.set_mode("none")
			return
		super().keyPressEvent(event)


	def _color_at_position(self, scene_pos: QPointF) -> Optional[QColor]:
		"""Get color from the image at the given scene position."""
		# Determine which image to pick from based on current display
		if self._show_working and self._working_pixmap is not None:
			# Pick from the working image
			source_image = self._working_pixmap.toImage()
		else:
			# Pick from the base image
			source_image = self._base_pixmap.toImage()
		
		if source_image is None:
			return None
		
		# Convert scene position to image coordinates
		x = int(scene_pos.x())
		y = int(scene_pos.y())
		
		# Check bounds
		if x < 0 or y < 0 or x >= source_image.width() or y >= source_image.height():
			return None
		
		# Get the color at the pixel using pixel() method for more reliable color extraction
		pixel_value = source_image.pixel(x, y)
		
		# Extract RGB values from the pixel value
		red = (pixel_value >> 16) & 0xFF
		green = (pixel_value >> 8) & 0xFF
		blue = pixel_value & 0xFF
		alpha = (pixel_value >> 24) & 0xFF
		
		# Create QColor from extracted values
		return QColor(red, green, blue, alpha)



	def _update_overlay(self) -> None:
		# If working image is enabled, show the working image
		if self._show_working and self._working_pixmap is not None:
			self._pix_item.setPixmap(self._working_pixmap)
			self._view.viewport().update()
		elif self._base_pixmap is not None:
			self._pix_item.setPixmap(self._base_pixmap)
			self._view.viewport().update()

	def _on_base_image_changed(self) -> None:
		"""Handle base image changes from AppState."""
		base_image = self._app_state.base_image
		if base_image is not None:
			# Ensure we have a QImage
			from utils.qt_image import ensure_qimage
			qimg = ensure_qimage(base_image)
			self._base_pixmap = QPixmap.fromImage(qimg)
			self._update_overlay()
			# Fit the image to the view when base image changes
			self._view.fit_in_view()

	def _on_working_image_changed(self) -> None:
		"""Handle working image changes from AppState."""
		working_image = self._app_state.working_image
		if working_image is not None:
			# Ensure we have a QImage
			from utils.qt_image import ensure_qimage
			qimg = ensure_qimage(working_image)
			self._working_pixmap = QPixmap.fromImage(qimg)
			self._update_overlay()

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

	def mousePressEvent(self, event: QMouseEvent) -> None:
		"""Handle mouse press events."""
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event: QMouseEvent) -> None:
		"""Handle mouse move events."""
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event: QMouseEvent) -> None:
		"""Handle mouse release events."""
		super().mouseReleaseEvent(event)


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
