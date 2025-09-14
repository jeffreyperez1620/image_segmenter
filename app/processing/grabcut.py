from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2 as cv


def apply_grabcut(
	bgr_image: np.ndarray,
	rect_xywh: Optional[Tuple[int, int, int, int]] = None,
	init_mask: Optional[np.ndarray] = None,
	iterations: int = 10,
) -> np.ndarray:
	"""
	Run GrabCut on a BGR image.

	Parameters
	----------
	bgr_image: np.ndarray
		Input image in BGR color order, dtype=uint8, shape (H, W, 3)
	rect_xywh: (x, y, w, h) or None
		Rectangle to initialize GrabCut. If None, mask must be provided.
	init_mask: np.ndarray or None
		Optional initial mask with values in {0,1,2,3} for {BGD,FGD,PR_BGD,PR_FGD}.
	iterations: int
		Number of iterations to run.

	Returns
	-------
	np.ndarray
		Result mask with values 0=BGD, 1=FGD, 2=PR_BGD, 3=PR_FGD, dtype=uint8
	"""
	if bgr_image.dtype != np.uint8 or bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
		raise ValueError("bgr_image must be HxWx3 uint8 BGR")

	h, w = bgr_image.shape[:2]
	mask = np.zeros((h, w), dtype=np.uint8)
	if init_mask is not None:
		if init_mask.shape != (h, w):
			raise ValueError("init_mask shape must match image")
		mask[:] = init_mask.astype(np.uint8)
		# Ensure mask has valid values (0, 1, 2, 3)
		mask = np.clip(mask, 0, 3).astype(np.uint8)

	bgd_model = np.zeros((1, 65), np.float64)
	fgd_model = np.zeros((1, 65), np.float64)

	if rect_xywh is not None:
		x, y, rw, rh = rect_xywh
		r = (int(x), int(y), int(rw), int(rh))
		cv.grabCut(bgr_image, mask, r, bgd_model, fgd_model, iterations, cv.GC_INIT_WITH_RECT)
	elif init_mask is not None:
		cv.grabCut(bgr_image, mask, None, bgd_model, fgd_model, iterations, cv.GC_INIT_WITH_MASK)
	else:
		raise ValueError("Either rect_xywh or init_mask must be provided")

	return mask
