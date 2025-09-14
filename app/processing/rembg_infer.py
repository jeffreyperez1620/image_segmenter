from __future__ import annotations

from typing import Optional, Dict, Tuple

import io
import threading
import numpy as np
from PIL import Image
from rembg import remove, new_session

# Cache rembg sessions by model id
_sessions_lock = threading.Lock()
_sessions: Dict[str, object] = {}


def _get_session(model: str) -> object:
	with _sessions_lock:
		sess = _sessions.get(model)
		if sess is None:
			sess = new_session(model)
			_sessions[model] = sess
		return sess


def _ensure_rgba(arr: np.ndarray) -> np.ndarray:
	if arr.ndim != 3:
		raise ValueError("Unexpected rembg array shape")
	if arr.shape[2] == 4:
		return arr
	if arr.shape[2] == 3:
		alpha = np.where((arr[:, :, 0] | arr[:, :, 1] | arr[:, :, 2]) > 0, 255, 0).astype(np.uint8)
		return np.dstack([arr, alpha])
	raise ValueError("Unexpected rembg channel count")


def _resize_rgba(rgba: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
	th, tw = target_hw
	if rgba.shape[0] == th and rgba.shape[1] == tw:
		return rgba
	img = Image.fromarray(rgba, mode="RGBA")
	# Use nearest neighbor interpolation to preserve sharp edges and avoid feathering
	img = img.resize((tw, th), resample=Image.NEAREST)
	return np.array(img, dtype=np.uint8)


def rembg_remove_bgr_to_rgba(bgr: np.ndarray, model: Optional[str] = None, target_hw: Optional[Tuple[int, int]] = None, sharp_edges: bool = True) -> np.ndarray:
	if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
		raise ValueError("bgr must be HxWx3 uint8")
	rgb = bgr[:, :, ::-1]
	if model:
		session = _get_session(model)
		out = remove(rgb, session=session)
	else:
		out = remove(rgb)
	# Convert to RGBA ndarray
	if isinstance(out, np.ndarray):
		rgba = _ensure_rgba(out)
	elif isinstance(out, (bytes, bytearray)):
		img = Image.open(io.BytesIO(out)).convert("RGBA")
		rgba = np.array(img, dtype=np.uint8)
	else:
		raise ValueError("Unexpected rembg output type")
	# Normalize size to match input or provided target
	if target_hw is None:
		target_hw = (rgb.shape[0], rgb.shape[1])
	rgba = _resize_rgba(rgba, target_hw)
	
	# Convert smooth alpha to sharp binary alpha if requested
	if sharp_edges:
		rgba = _make_alpha_sharp(rgba)
	
	return rgba


def _make_alpha_sharp(rgba: np.ndarray) -> np.ndarray:
	"""
	Convert smooth alpha channel to sharp binary alpha to eliminate feathering.
	Uses Otsu's thresholding to find the optimal cutoff point.
	"""
	import cv2 as cv
	
	alpha = rgba[:, :, 3]
	
	# Use Otsu's method to find optimal threshold
	_, binary_alpha = cv.threshold(alpha, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	
	# Apply morphological operations to clean up the binary mask
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
	binary_alpha = cv.morphologyEx(binary_alpha, cv.MORPH_CLOSE, kernel)
	binary_alpha = cv.morphologyEx(binary_alpha, cv.MORPH_OPEN, kernel)
	
	# Create result with sharp alpha
	result = rgba.copy()
	result[:, :, 3] = binary_alpha
	
	return result
