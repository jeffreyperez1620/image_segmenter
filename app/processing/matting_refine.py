from __future__ import annotations

from typing import Optional, Tuple
import warnings

import numpy as np

try:
	from pymatting import estimate_alpha_cf
except Exception:  # pragma: no cover
	from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf  # type: ignore


def build_trimap_from_alpha_and_strokes(
	rgba_or_rgb: np.ndarray,
	alpha_init: np.ndarray,
	user_mask: Optional[np.ndarray],
	fg_threshold: int = 220,
	bg_threshold: int = 20,
) -> np.ndarray:
	"""
	Build a trimap (encoded as 0=BG, 0.5=Unknown, 1=FG) as float64.
	"""
	h, w = alpha_init.shape
	trimap = np.full((h, w), 0.5, dtype=np.float64)
	
	# Adaptive thresholds based on alpha distribution
	alpha_min, alpha_max = alpha_init.min(), alpha_init.max()
	if alpha_max - alpha_min > 50:  # If there's good contrast
		trimap[alpha_init >= fg_threshold] = 1.0
		trimap[alpha_init <= bg_threshold] = 0.0
	else:
		# Use percentiles for better adaptation
		trimap[alpha_init >= np.percentile(alpha_init, 80)] = 1.0
		trimap[alpha_init <= np.percentile(alpha_init, 20)] = 0.0
	
	# Apply user mask constraints (these override the above)
	if user_mask is not None:
		trimap[user_mask == 1] = 1.0
		trimap[user_mask == 2] = 0.0
	
	# Ensure we have both foreground and background regions
	fg_pixels = np.sum(trimap == 1.0)
	bg_pixels = np.sum(trimap == 0.0)
	
	if fg_pixels == 0 or bg_pixels == 0:
		# Fallback to simple thresholds if no good regions found
		trimap[alpha_init >= 128] = 1.0
		trimap[alpha_init < 128] = 0.0
		if user_mask is not None:
			trimap[user_mask == 1] = 1.0
			trimap[user_mask == 2] = 0.0
	
	return trimap


def simple_alpha_refine(
	rgb: np.ndarray,
	alpha_init: np.ndarray,
	user_mask: Optional[np.ndarray],
) -> np.ndarray:
	"""
	Simple alpha refinement using morphological operations and smoothing.
	This is a fallback when the advanced matting fails.
	"""
	import cv2 as cv
	
	# Create a binary mask from the initial alpha
	binary_mask = (alpha_init > 128).astype(np.uint8) * 255
	
	# Apply morphological operations to clean up the mask
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
	binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_CLOSE, kernel)
	binary_mask = cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel)
	
	# Apply Gaussian blur to create smooth alpha
	alpha_smooth = cv.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 1.0)
	
	# Apply user mask constraints
	if user_mask is not None:
		alpha_smooth[user_mask == 1] = 255.0
		alpha_smooth[user_mask == 2] = 0.0
	
	return np.clip(alpha_smooth, 0, 255).astype(np.uint8)


def refine_alpha_portrait(
	rgb: np.ndarray,
	alpha_init: np.ndarray,
	user_mask: Optional[np.ndarray],
) -> np.ndarray:
	"""
	Refine initial alpha using closed-form matting guided by a trimap.
	Returns uint8 alpha (0..255)
	"""
	if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
		raise ValueError("rgb must be HxWx3 uint8")
	if alpha_init.dtype != np.uint8 or alpha_init.shape[:2] != rgb.shape[:2]:
		raise ValueError("alpha_init must be HxW uint8 matching image")

	# Check if image is too large for efficient processing
	h, w = rgb.shape[:2]
	max_dimension = 1024  # Limit to 1024 pixels for better convergence
	
	if h > max_dimension or w > max_dimension:
		# Downsample for processing
		import cv2 as cv
		scale = min(max_dimension / h, max_dimension / w)
		new_h, new_w = int(h * scale), int(w * scale)
		
		rgb_small = cv.resize(rgb, (new_w, new_h), interpolation=cv.INTER_AREA)
		alpha_small = cv.resize(alpha_init, (new_w, new_h), interpolation=cv.INTER_AREA)
		user_mask_small = None
		if user_mask is not None:
			user_mask_small = cv.resize(user_mask, (new_w, new_h), interpolation=cv.INTER_NEAREST)
		
		# Process smaller image
		alpha_refined_small = refine_alpha_portrait_small(rgb_small, alpha_small, user_mask_small)
		
		# Upsample result back to original size
		alpha_refined = cv.resize(alpha_refined_small, (w, h), interpolation=cv.INTER_LINEAR)
		return alpha_refined
	else:
		return refine_alpha_portrait_small(rgb, alpha_init, user_mask)


def refine_alpha_portrait_small(
	rgb: np.ndarray,
	alpha_init: np.ndarray,
	user_mask: Optional[np.ndarray],
) -> np.ndarray:
	"""
	Refine initial alpha using closed-form matting guided by a trimap.
	This version is optimized for smaller images.
	"""
	trimap = build_trimap_from_alpha_and_strokes(rgb, alpha_init, user_mask)
	# Convert to float64 [0,1]
	rgb_f = (rgb.astype(np.float64) / 255.0)
	
	# estimate_alpha_cf expects image float64 and trimap in [0,1]
	# Try with basic parameters first
	try:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			alpha_refined = estimate_alpha_cf(rgb_f, trimap)
	except Exception as e:
		# If basic approach fails, try with laplacian parameters
		try:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				alpha_refined = estimate_alpha_cf(
					rgb_f, 
					trimap,
					laplacian_kwargs={"epsilon": 1e-6, "radius": 1}
				)
		except Exception as e2:
			# If still failing, use simple refinement as fallback
			print(f"Warning: Advanced portrait matting failed to converge. Using simple refinement. Error: {e2}")
			return simple_alpha_refine(rgb, alpha_init, user_mask)
	
	alpha_u8 = np.clip((alpha_refined * 255.0 + 0.5).astype(np.uint8), 0, 255)
	# Enforce strokes again
	if user_mask is not None:
		alpha_u8[user_mask == 1] = 255
		alpha_u8[user_mask == 2] = 0
	return alpha_u8
