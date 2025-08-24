from __future__ import annotations

from typing import Tuple

import numpy as np
from PySide6.QtGui import QImage


def qimage_to_numpy_bgr(image: QImage) -> np.ndarray:
	# Ensure 4-channel RGBA for consistent memory layout
	if image.format() != QImage.Format.Format_RGBA8888:
		img = image.convertToFormat(QImage.Format.Format_RGBA8888)
	else:
		img = image

	w = img.width()
	h = img.height()
	ptr = img.constBits()
	arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
	# Convert RGBA to BGR (drop alpha)
	bgr = arr[:, :, :3][:, :, ::-1].copy()
	return bgr


def numpy_rgba_to_qimage(rgba: np.ndarray) -> QImage:
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	h, w = rgba.shape[:2]
	# QImage expects bytes in RGBA byte order for Format_RGBA8888
	img = QImage(rgba.data, w, h, QImage.Format.Format_RGBA8888)
	# Deep copy to detach from numpy buffer
	return img.copy()


def composite_foreground_over_transparent(bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
	"""Compose BGR foreground over transparent background using mask01 in {0,1}. Returns RGBA."""
	if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
		raise ValueError("bgr must be HxWx3 uint8")
	if mask01.dtype != np.uint8:
		mask01 = (mask01 > 0).astype(np.uint8)
	alpha = (mask01 * 255).astype(np.uint8)
	rgb = bgr[:, :, ::-1]
	rgba = np.dstack([rgb, alpha])
	return rgba
