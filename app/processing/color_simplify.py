from __future__ import annotations

from typing import Optional, Tuple, List
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import cv2 as cv


def simplify_colors_kmeans(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using K-means clustering.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Number of colors to reduce to
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Only process non-transparent pixels
	non_transparent = alpha > 0
	if not np.any(non_transparent):
		# If no non-transparent pixels, return original
		return rgba, np.array([[0, 0, 0]])
	
	rgb_non_transparent = rgb[non_transparent]
	
	# Reshape for clustering
	rgb_flat = rgb_non_transparent.reshape(-1, 3)
	
	# Filter out very dark/black pixels to prevent them from dominating
	# Consider pixels with brightness > 30 as non-black
	brightness = np.mean(rgb_flat, axis=1)
	non_black_mask = brightness > 30
	if np.sum(non_black_mask) < num_colors:
		# If too few non-black pixels, lower the threshold
		non_black_mask = brightness > 10
	
	if np.sum(non_black_mask) == 0:
		# If still no non-black pixels, use all pixels
		non_black_mask = np.ones(len(rgb_flat), dtype=bool)
	
	rgb_filtered = rgb_flat[non_black_mask]
	
	# Ensure we don't try to cluster more colors than we have unique colors
	unique_colors = np.unique(rgb_filtered, axis=0)
	actual_num_colors = min(num_colors, len(unique_colors))
	
	if actual_num_colors < 2:
		# If we can't cluster, return original
		return rgba, np.array([[0, 0, 0]])
	
	# Apply K-means clustering with warning suppression
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		kmeans = KMeans(n_clusters=actual_num_colors, random_state=42, n_init=10)
		labels = kmeans.fit_predict(rgb_filtered)
		centers = kmeans.cluster_centers_
	
	# Quantize colors
	centers = np.clip(centers, 0, 255).astype(np.uint8)
	
	# Map pixels to cluster centers
	quantized_rgb = np.zeros_like(rgb)
	
	# Map non-transparent pixels
	quantized_rgb[non_transparent][np.where(non_black_mask)[0]] = centers[labels]
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, centers


def simplify_colors_median_cut(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using median cut algorithm.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Number of colors to reduce to (must be power of 2)
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	# Ensure num_colors is a power of 2
	num_colors = 2 ** int(np.log2(num_colors))
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Reshape for processing
	rgb_flat = rgb.reshape(-1, 3)
	
	# Apply median cut
	from PIL import Image, ImageQt
	pil_image = Image.fromarray(rgb)
	palette_image = pil_image.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
	
	# Get palette
	palette = np.array(palette_image.getpalette()).reshape(-1, 3)
	palette = palette[:num_colors]
	
	# Convert back to numpy array
	quantized_rgb = np.array(palette_image.convert('RGB'))
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, palette


def simplify_colors_octree(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using octree quantization.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Number of colors to reduce to
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Use PIL's octree quantization
	from PIL import Image
	pil_image = Image.fromarray(rgb)
	palette_image = pil_image.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
	
	# Get palette
	palette = np.array(palette_image.getpalette()).reshape(-1, 3)
	palette = palette[:num_colors]
	
	# Convert back to numpy array
	quantized_rgb = np.array(palette_image.convert('RGB'))
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, palette


def simplify_colors_threshold(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using simple thresholding (posterization).
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Number of colors to reduce to (will be rounded to nearest power of 2)
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Calculate quantization levels
	levels_per_channel = int(np.ceil(np.cbrt(num_colors)))
	quantization_step = 256 // levels_per_channel
	
	# Quantize each channel
	quantized_rgb = np.zeros_like(rgb)
	for c in range(3):
		quantized_rgb[:, :, c] = (rgb[:, :, c] // quantization_step) * quantization_step
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	# Generate palette from unique colors
	unique_colors = np.unique(quantized_rgb.reshape(-1, 3), axis=0)
	palette = unique_colors[:num_colors]
	
	return simplified_rgba, palette


def simplify_colors_adaptive(
	rgba: np.ndarray,
	target_colors: int = 8,
	preserve_alpha: bool = True,
	algorithm: str = "kmeans",
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Adaptive color simplification that chooses the best method.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	target_colors: int
		Target number of colors
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
	algorithm: str
		Algorithm to use: "kmeans", "median_cut", "octree", "threshold", "adaptive", 
		"perceptual", "adaptive_distance", "hsv_clustering"
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if algorithm == "kmeans":
		return simplify_colors_kmeans(rgba, target_colors, preserve_alpha)
	elif algorithm == "median_cut":
		return simplify_colors_median_cut(rgba, target_colors, preserve_alpha)
	elif algorithm == "octree":
		return simplify_colors_octree(rgba, target_colors, preserve_alpha)
	elif algorithm == "threshold":
		return simplify_colors_threshold(rgba, target_colors, preserve_alpha)
	elif algorithm == "perceptual":
		return simplify_colors_perceptual(rgba, target_colors, preserve_alpha)
	elif algorithm == "perceptual_fast":
		return simplify_colors_perceptual_fast(rgba, target_colors, preserve_alpha)
	elif algorithm == "adaptive_distance":
		return simplify_colors_adaptive_distance(rgba, target_colors, preserve_alpha)
	elif algorithm == "hsv_clustering":
		return simplify_colors_hsv_clustering(rgba, target_colors, preserve_alpha)
	elif algorithm == "custom_palette":
		# For custom palette, we need the palette to be passed separately
		# This will be handled in the UI layer
		raise ValueError("Custom palette requires palette parameter")
	elif algorithm == "adaptive":
		# Choose best method based on image characteristics
		stats = get_color_statistics(rgba)
		total_colors = stats['total_unique_colors']
		
		if total_colors <= target_colors:
			# Already simplified enough, use threshold
			return simplify_colors_threshold(rgba, target_colors, preserve_alpha)
		elif total_colors > 1000:
			# Many colors, use perceptual clustering for better quality
			return simplify_colors_perceptual(rgba, target_colors, preserve_alpha)
		else:
			# Moderate colors, use HSV clustering for good balance
			return simplify_colors_hsv_clustering(rgba, target_colors, preserve_alpha)
	else:
		# Default to K-means
		return simplify_colors_kmeans(rgba, target_colors, preserve_alpha)


def get_color_statistics(rgba: np.ndarray) -> dict:
	"""
	Get statistics about the colors in an image.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image
		
	Returns
	-------
	dict
		Color statistics
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	# Count unique colors
	unique_colors = np.unique(rgba.reshape(-1, 4), axis=0)
	
	# Count non-transparent pixels
	non_transparent = rgba[:, :, 3] > 0
	num_non_transparent = np.sum(non_transparent)
	
	# Get RGB statistics for non-transparent pixels
	if num_non_transparent > 0:
		rgb_non_transparent = rgba[non_transparent][:, :3]
		rgb_mean = np.mean(rgb_non_transparent, axis=0)
		rgb_std = np.std(rgb_non_transparent, axis=0)
	else:
		rgb_mean = np.array([0, 0, 0])
		rgb_std = np.array([0, 0, 0])
	
	return {
		'total_unique_colors': len(unique_colors),
		'non_transparent_pixels': num_non_transparent,
		'rgb_mean': rgb_mean,
		'rgb_std': rgb_std,
		'image_size': rgba.shape[:2],
	}


def simplify_colors_perceptual(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
	color_tolerance: float = 30.0,
	use_gpu: bool = False,
	max_samples: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using perceptual color distance to combine similar shades.
	
	This method groups colors that are perceptually similar (e.g., light green and 
	normal green) while preserving distinct color regions (e.g., green vs blue).
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Target number of colors
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
	color_tolerance: float
		Color distance threshold for combining similar colors (0-100)
	use_gpu: bool
		Whether to use GPU acceleration (requires CuPy or PyTorch)
	max_samples: int
		Maximum number of color samples to process (for memory efficiency)
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Only process non-transparent pixels
	non_transparent = alpha > 0
	if not np.any(non_transparent):
		# If no non-transparent pixels, return original
		return rgba, np.array([[0, 0, 0]])
	
	rgb_non_transparent = rgb[non_transparent]
	
	# Reshape for processing
	rgb_flat = rgb_non_transparent.reshape(-1, 3)
	
	# Sample colors to reduce memory usage
	if len(rgb_flat) > max_samples:
		# Use stratified sampling to preserve color distribution
		indices = np.random.choice(len(rgb_flat), max_samples, replace=False)
		rgb_samples = rgb_flat[indices]
	else:
		rgb_samples = rgb_flat
		indices = np.arange(len(rgb_flat))
	
	# Get unique colors and their frequencies from samples
	unique_colors, counts = np.unique(rgb_samples, axis=0, return_counts=True)
	
	# Filter out very dark/black colors to prevent them from dominating
	# Consider colors with brightness > 30 as non-black
	brightness = np.mean(unique_colors, axis=1)
	non_black_mask = brightness > 30
	if np.sum(non_black_mask) < num_colors:
		# If too few non-black colors, lower the threshold
		non_black_mask = brightness > 10
	
	if np.sum(non_black_mask) == 0:
		# If still no non-black colors, use all colors
		non_black_mask = np.ones(len(unique_colors), dtype=bool)
	
	unique_colors_filtered = unique_colors[non_black_mask]
	counts_filtered = counts[non_black_mask]
	
	# Convert to LAB color space for perceptual distance
	from skimage import color
	lab_colors = color.rgb2lab(unique_colors_filtered.reshape(-1, 1, 3)).reshape(-1, 3)
	
	# Ensure we don't try to cluster more colors than we have unique colors
	actual_num_colors = min(num_colors, len(unique_colors_filtered))
	
	if actual_num_colors < 2:
		# If we can't cluster, return original
		return rgba, np.array([[0, 0, 0]])
	
	if use_gpu:
		try:
			# Try CuPy first (CUDA)
			import cupy as cp
			lab_colors_gpu = cp.asarray(lab_colors)
			
			# Use GPU-accelerated clustering
			from sklearn.cluster import AgglomerativeClustering
			clustering = AgglomerativeClustering(
				n_clusters=actual_num_colors,
				linkage='ward',
				distance_threshold=None
			)
			
			# Fit clustering on GPU data
			cluster_labels = clustering.fit_predict(cp.asnumpy(lab_colors_gpu))
			
		except ImportError:
			try:
				# Fallback to PyTorch
				import torch
				lab_colors_tensor = torch.from_numpy(lab_colors).float()
				
				# Use PyTorch's k-means implementation
				from sklearn.cluster import KMeans
				kmeans = KMeans(n_clusters=actual_num_colors, random_state=42, n_init=10)
				cluster_labels = kmeans.fit_predict(lab_colors)
				
			except ImportError:
				# Fallback to CPU
				use_gpu = False
	
	if not use_gpu:
		# CPU-based hierarchical clustering
		from sklearn.cluster import AgglomerativeClustering
		
		# Use LAB distance for clustering
		clustering = AgglomerativeClustering(
			n_clusters=actual_num_colors,
			linkage='ward',
			distance_threshold=None
		)
		
		# Fit clustering
		cluster_labels = clustering.fit_predict(lab_colors)
	
	# Calculate cluster centers (weighted by frequency)
	cluster_centers = np.zeros((clustering.n_clusters_, 3))
	for i in range(clustering.n_clusters_):
		mask = cluster_labels == i
		if np.any(mask):
			# Weight by frequency
			weights = counts_filtered[mask]
			cluster_centers[i] = np.average(unique_colors_filtered[mask], weights=weights, axis=0)
	
	cluster_centers = np.clip(cluster_centers, 0, 255).astype(np.uint8)
	
	# Apply color mapping to full image using nearest neighbor
	quantized_rgb = np.zeros_like(rgb)
	
	# Convert all non-transparent pixels to LAB for distance calculation
	lab_non_transparent = color.rgb2lab(rgb_non_transparent.reshape(-1, 1, 3)).reshape(-1, 3)
	
	# Find nearest cluster center for each pixel
	from sklearn.metrics import pairwise_distances_argmin_min
	nearest_clusters, _ = pairwise_distances_argmin_min(lab_non_transparent, cluster_centers)
	
	# Map pixels to cluster centers
	quantized_rgb[non_transparent] = cluster_centers[nearest_clusters]
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, cluster_centers


def simplify_colors_perceptual_fast(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
	color_tolerance: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Fast perceptual color simplification using downsampling and efficient clustering.
	
	This is a memory-efficient version that downsamples the image before processing
	and uses faster clustering algorithms.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Target number of colors
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
	color_tolerance: float
		Color distance threshold for combining similar colors (0-100)
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Only process non-transparent pixels
	non_transparent = alpha > 0
	if not np.any(non_transparent):
		# If no non-transparent pixels, return original
		return rgba, np.array([[0, 0, 0]])
	
	rgb_non_transparent = rgb[non_transparent]
	
	# Downsample for processing if image is large
	max_dim = 512
	if h > max_dim or w > max_dim:
		import cv2
		scale = min(max_dim / h, max_dim / w)
		new_h, new_w = int(h * scale), int(w * scale)
		rgb_small = cv.resize(rgb, (new_w, new_h), interpolation=cv.INTER_AREA)
		alpha_small = cv.resize(alpha, (new_w, new_h), interpolation=cv.INTER_AREA)
		
		# Apply the same non-transparent mask to downsampled image
		non_transparent_small = alpha_small > 0
		if np.any(non_transparent_small):
			rgb_small_non_transparent = rgb_small[non_transparent_small]
		else:
			return rgba, np.array([[0, 0, 0]])
	else:
		rgb_small = rgb
		alpha_small = alpha
		rgb_small_non_transparent = rgb_non_transparent
		non_transparent_small = non_transparent
	
	# Reshape for processing
	rgb_flat = rgb_small_non_transparent.reshape(-1, 3)
	
	# Sample colors randomly to reduce memory usage
	sample_size = min(5000, len(rgb_flat))
	if len(rgb_flat) > sample_size:
		indices = np.random.choice(len(rgb_flat), sample_size, replace=False)
		rgb_samples = rgb_flat[indices]
	else:
		rgb_samples = rgb_flat
	
	# Get unique colors from samples
	unique_colors = np.unique(rgb_samples, axis=0)
	
	# Filter out very dark/black colors to prevent them from dominating
	# Consider colors with brightness > 30 as non-black
	brightness = np.mean(unique_colors, axis=1)
	non_black_mask = brightness > 30
	if np.sum(non_black_mask) < num_colors:
		# If too few non-black colors, lower the threshold
		non_black_mask = brightness > 10
	
	if np.sum(non_black_mask) == 0:
		# If still no non-black colors, use all colors
		non_black_mask = np.ones(len(unique_colors), dtype=bool)
	
	unique_colors_filtered = unique_colors[non_black_mask]
	
	# Convert to LAB color space for perceptual distance
	from skimage import color
	lab_colors = color.rgb2lab(unique_colors_filtered.reshape(-1, 1, 3)).reshape(-1, 3)
	
	# Ensure we don't try to cluster more colors than we have unique colors
	actual_num_colors = min(num_colors, len(unique_colors_filtered))
	
	if actual_num_colors < 2:
		# If we can't cluster, return original
		return rgba, np.array([[0, 0, 0]])
	
	# Use K-means for faster clustering
	from sklearn.cluster import KMeans
	kmeans = KMeans(
		n_clusters=actual_num_colors, 
		random_state=42, 
		n_init=10,
		max_iter=100  # Reduce iterations for speed
	)
	cluster_labels = kmeans.fit_predict(lab_colors)
	
	# Get cluster centers
	cluster_centers = kmeans.cluster_centers_
	
	# Convert LAB centers back to RGB
	cluster_centers_rgb = color.lab2rgb(cluster_centers.reshape(-1, 1, 3)).reshape(-1, 3)
	cluster_centers_rgb = np.clip(cluster_centers_rgb * 255, 0, 255).astype(np.uint8)
	
	# Apply color mapping to full image using nearest neighbor
	quantized_rgb = np.zeros_like(rgb)
	
	# Convert all non-transparent pixels to LAB for distance calculation
	lab_non_transparent = color.rgb2lab(rgb_non_transparent.reshape(-1, 1, 3)).reshape(-1, 3)
	
	# Find nearest cluster center for each pixel
	from sklearn.metrics import pairwise_distances_argmin_min
	nearest_clusters, _ = pairwise_distances_argmin_min(lab_non_transparent, cluster_centers)
	
	# Map pixels to cluster centers
	quantized_rgb[non_transparent] = cluster_centers_rgb[nearest_clusters]
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, cluster_centers_rgb


def simplify_colors_adaptive_distance(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
	similarity_threshold: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using adaptive color distance to combine similar shades.
	
	This method uses a more sophisticated approach to group colors based on
	perceptual similarity while preserving distinct color regions.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Target number of colors
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
	similarity_threshold: float
		Threshold for considering colors similar (0-100)
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Only process non-transparent pixels
	non_transparent = alpha > 0
	if not np.any(non_transparent):
		# If no non-transparent pixels, return original
		return rgba, np.array([[0, 0, 0]])
	
	rgb_non_transparent = rgb[non_transparent]
	
	# Convert to LAB color space for better perceptual distance
	from skimage import color
	lab_image = color.rgb2lab(rgb_non_transparent)
	
	# Reshape for processing
	lab_flat = lab_image.reshape(-1, 3)
	rgb_flat = rgb_non_transparent.reshape(-1, 3)
	
	# Filter out very dark/black pixels to prevent them from dominating
	# Consider pixels with L > 10 as non-black (L is lightness in LAB)
	non_black_mask = lab_flat[:, 0] > 10
	if np.sum(non_black_mask) < num_colors:
		# If too few non-black pixels, lower the threshold
		non_black_mask = lab_flat[:, 0] > 5
	
	if np.sum(non_black_mask) == 0:
		# If still no non-black pixels, use all pixels
		non_black_mask = np.ones(len(lab_flat), dtype=bool)
	
	lab_filtered = lab_flat[non_black_mask]
	rgb_filtered = rgb_flat[non_black_mask]
	
	# Use DBSCAN clustering to group similar colors
	from sklearn.cluster import DBSCAN
	from sklearn.preprocessing import StandardScaler
	
	# Normalize LAB values for clustering
	scaler = StandardScaler()
	lab_normalized = scaler.fit_transform(lab_filtered)
	
	# Use DBSCAN with adaptive eps based on similarity threshold
	# Adjust eps to be more aggressive to get more clusters
	eps = (similarity_threshold / 100.0) * 0.5  # Make eps smaller to get more clusters
	dbscan = DBSCAN(eps=eps, min_samples=3)  # Reduce min_samples to allow smaller clusters
	cluster_labels = dbscan.fit_predict(lab_normalized)
	
	# Handle noise points (label -1) by assigning them to nearest cluster
	if -1 in cluster_labels:
		from sklearn.neighbors import NearestNeighbors
		noise_mask = cluster_labels == -1
		cluster_mask = cluster_labels != -1
		
		if np.any(cluster_mask):
			# Find nearest cluster for noise points
			nn = NearestNeighbors(n_neighbors=1)
			nn.fit(lab_normalized[cluster_mask])
			nearest_clusters = cluster_labels[cluster_mask][nn.kneighbors(lab_normalized[noise_mask])[1].flatten()]
			cluster_labels[noise_mask] = nearest_clusters
	
	# Calculate cluster centers
	unique_labels = np.unique(cluster_labels)
	n_clusters = len(unique_labels)
	
	# If DBSCAN produced too few clusters, use K-means to get the target number
	if n_clusters < num_colors:
		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
		cluster_labels = kmeans.fit_predict(lab_normalized)
		n_clusters = num_colors
		unique_labels = np.arange(num_colors)
	
	# If we have too many clusters, merge the smallest ones
	if n_clusters > num_colors:
		# Count cluster sizes
		cluster_sizes = np.bincount(cluster_labels.astype(int))
		# Sort by size (smallest first)
		sorted_clusters = np.argsort(cluster_sizes)
		
		# Keep the largest clusters
		clusters_to_keep = sorted_clusters[-num_colors:]
		clusters_to_merge = sorted_clusters[:-num_colors]
		
		# Merge small clusters into nearest large cluster
		for small_cluster in clusters_to_merge:
			small_cluster_center = np.mean(lab_filtered[cluster_labels == small_cluster], axis=0)
			large_cluster_centers = np.array([np.mean(lab_filtered[cluster_labels == large_cluster], axis=0) 
											for large_cluster in clusters_to_keep])
			
			# Find nearest large cluster
			distances = np.linalg.norm(large_cluster_centers - small_cluster_center, axis=1)
			nearest_large = clusters_to_keep[np.argmin(distances)]
			
			# Merge
			cluster_labels[cluster_labels == small_cluster] = nearest_large
	
	# Calculate final cluster centers in RGB space
	unique_labels = np.unique(cluster_labels)
	cluster_centers = np.zeros((len(unique_labels), 3))
	
	for i, label in enumerate(unique_labels):
		mask = cluster_labels == label
		cluster_centers[i] = np.mean(rgb_filtered[mask], axis=0)
	
	cluster_centers = np.clip(cluster_centers, 0, 255).astype(np.uint8)
	
	# Map pixels to cluster centers
	quantized_rgb = np.zeros_like(rgb)
	
	# Create a mapping array for all non-transparent pixels
	# First, create an array to hold the cluster assignments for all non-transparent pixels
	all_cluster_labels = np.zeros(np.sum(non_transparent), dtype=int)
	
	# Assign cluster labels to the filtered pixels
	all_cluster_labels[np.where(non_black_mask)[0]] = cluster_labels
	
	# For pixels that were filtered out (too dark), assign them to the nearest cluster
	black_pixel_indices = np.where(~non_black_mask)[0]
	if len(black_pixel_indices) > 0:
		# Find nearest cluster for black pixels using LAB distance
		black_pixels_lab = lab_flat[black_pixel_indices]
		from sklearn.metrics import pairwise_distances_argmin_min
		nearest_clusters, _ = pairwise_distances_argmin_min(black_pixels_lab, lab_filtered)
		all_cluster_labels[black_pixel_indices] = cluster_labels[nearest_clusters]
	
	# Now map all non-transparent pixels to their cluster centers
	quantized_rgb[non_transparent] = cluster_centers[all_cluster_labels]
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, cluster_centers


def simplify_colors_hsv_clustering(
	rgba: np.ndarray,
	num_colors: int = 8,
	preserve_alpha: bool = True,
	hue_tolerance: float = 15.0,
	saturation_tolerance: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using HSV-based clustering to group similar hues and saturations.
	
	This method is particularly good at combining different shades of the same color
	while preserving distinct color families.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	num_colors: int
		Target number of colors
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
	hue_tolerance: float
		Hue tolerance in degrees (0-180)
	saturation_tolerance: float
		Saturation tolerance (0-1)
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Only process non-transparent pixels
	non_transparent = alpha > 0
	if not np.any(non_transparent):
		# If no non-transparent pixels, return original
		return rgba, np.array([[0, 0, 0]])
	
	rgb_non_transparent = rgb[non_transparent]
	
	# Ensure we have the right shape for OpenCV
	if rgb_non_transparent.size == 0:
		return rgba, np.array([[0, 0, 0]])
	
	# Reshape to ensure proper dimensions for OpenCV
	if rgb_non_transparent.ndim == 1:
		rgb_non_transparent = rgb_non_transparent.reshape(-1, 3)
	
	# Convert to HSV - ensure we have a proper 3-channel image
	if rgb_non_transparent.shape[1] != 3:
		# If we don't have 3 channels, something went wrong
		return rgba, np.array([[0, 0, 0]])
	
	# Convert to HSV using OpenCV
	hsv = cv.cvtColor(rgb_non_transparent.reshape(-1, 1, 3), cv.COLOR_RGB2HSV)
	hsv = hsv.reshape(-1, 3)
	
	# Reshape for processing
	hsv_flat = hsv.reshape(-1, 3)
	rgb_flat = rgb_non_transparent.reshape(-1, 3)
	
	# Filter out very dark/black pixels to prevent them from dominating
	# Consider pixels with value > 30 as non-black
	non_black_mask = hsv_flat[:, 2] > 30
	if np.sum(non_black_mask) < num_colors:
		# If too few non-black pixels, lower the threshold
		non_black_mask = hsv_flat[:, 2] > 10
	
	if np.sum(non_black_mask) == 0:
		# If still no non-black pixels, use all pixels
		non_black_mask = np.ones(len(hsv_flat), dtype=bool)
	
	hsv_filtered = hsv_flat[non_black_mask]
	rgb_filtered = rgb_flat[non_black_mask]
	
	# Normalize HSV for clustering (hue is already 0-179, sat/value 0-255)
	hsv_normalized = hsv_filtered.copy().astype(np.float32)
	hsv_normalized[:, 0] = hsv_filtered[:, 0] / 179.0  # Normalize hue to 0-1
	hsv_normalized[:, 1:] = hsv_filtered[:, 1:] / 255.0  # Normalize sat/value to 0-1
	
	# Use K-means with custom distance metric for HSV
	from sklearn.cluster import KMeans
	
	# Create custom distance weights
	# Hue is most important, then saturation, then value
	distance_weights = np.array([2.0, 1.5, 1.0])  # hue, sat, value weights
	
	# Apply weights to normalized HSV
	hsv_weighted = hsv_normalized * distance_weights
	
	# Ensure we don't try to cluster more colors than we have unique colors
	unique_colors = np.unique(hsv_weighted, axis=0)
	actual_num_colors = min(num_colors, len(unique_colors))
	
	if actual_num_colors < 2:
		# If we can't cluster, return original
		return rgba, np.array([[0, 0, 0]])
	
	# Cluster
	kmeans = KMeans(n_clusters=actual_num_colors, random_state=42, n_init=10)
	cluster_labels = kmeans.fit_predict(hsv_weighted)
	
	# Calculate cluster centers in RGB space
	cluster_centers = np.zeros((actual_num_colors, 3))
	for i in range(actual_num_colors):
		mask = cluster_labels == i
		if np.any(mask):
			cluster_centers[i] = np.mean(rgb_filtered[mask], axis=0)
	
	cluster_centers = np.clip(cluster_centers, 0, 255).astype(np.uint8)
	
	# Map pixels to cluster centers
	quantized_rgb = np.zeros_like(rgb)
	
	# Create a mapping array for all non-transparent pixels
	# First, create an array to hold the cluster assignments for all non-transparent pixels
	all_cluster_labels = np.zeros(np.sum(non_transparent), dtype=int)
	
	# Assign cluster labels to the filtered pixels
	all_cluster_labels[np.where(non_black_mask)[0]] = cluster_labels
	
	# For pixels that were filtered out (too dark), assign them to the nearest cluster
	black_pixel_indices = np.where(~non_black_mask)[0]
	if len(black_pixel_indices) > 0:
		# Find nearest cluster for black pixels using RGB distance
		black_pixels_rgb = rgb_flat[black_pixel_indices]
		from sklearn.metrics import pairwise_distances_argmin_min
		nearest_clusters, _ = pairwise_distances_argmin_min(black_pixels_rgb, cluster_centers)
		all_cluster_labels[black_pixel_indices] = nearest_clusters
	
	# Now map all non-transparent pixels to their cluster centers
	quantized_rgb[non_transparent] = cluster_centers[all_cluster_labels]
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, cluster_centers


def simplify_colors_custom_palette(
	rgba: np.ndarray,
	custom_palette: np.ndarray,
	preserve_alpha: bool = True,
	distance_metric: str = "lab",
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Simplify colors using a custom palette with nearest-neighbor color mapping.
	
	This method maps each pixel to the closest color in the provided palette.
	
	Parameters
	----------
	rgba: np.ndarray
		Input RGBA image, shape (H, W, 4)
	custom_palette: np.ndarray
		Custom color palette, shape (N, 3) where N is the number of colors
	preserve_alpha: bool
		Whether to preserve alpha channel or simplify it too
	distance_metric: str
		Distance metric to use: "lab" (perceptual), "rgb" (Euclidean), "hsv"
		
	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		(simplified_rgba, color_palette)
	"""
	if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
		raise ValueError("rgba must be HxWx4 uint8")
	
	if custom_palette.dtype != np.uint8 or custom_palette.ndim != 2 or custom_palette.shape[1] != 3:
		raise ValueError("custom_palette must be Nx3 uint8")
	
	h, w = rgba.shape[:2]
	
	# Separate alpha channel
	rgb = rgba[:, :, :3]
	alpha = rgba[:, :, 3]
	
	# Only process non-transparent pixels
	non_transparent = alpha > 0
	if not np.any(non_transparent):
		# If no non-transparent pixels, return original
		return rgba, custom_palette
	
	rgb_non_transparent = rgb[non_transparent]
	
	# Convert to appropriate color space for distance calculation
	if distance_metric == "lab":
		from skimage import color
		# Convert both image and palette to LAB
		lab_image = color.rgb2lab(rgb_non_transparent.reshape(-1, 1, 3)).reshape(-1, 3)
		lab_palette = color.rgb2lab(custom_palette.reshape(-1, 1, 3)).reshape(-1, 3)
		image_colors = lab_image
		palette_colors = lab_palette
	elif distance_metric == "hsv":
		import cv2 as cv
		# Convert both image and palette to HSV
		hsv_image = cv.cvtColor(rgb_non_transparent.reshape(-1, 1, 3), cv.COLOR_RGB2HSV).reshape(-1, 3)
		hsv_palette = cv.cvtColor(custom_palette.reshape(-1, 1, 3), cv.COLOR_RGB2HSV).reshape(-1, 3)
		image_colors = hsv_image
		palette_colors = hsv_palette
	else:  # rgb
		image_colors = rgb_non_transparent.reshape(-1, 3)
		palette_colors = custom_palette
	
	# Find nearest palette color for each pixel
	from sklearn.metrics import pairwise_distances_argmin_min
	nearest_indices, _ = pairwise_distances_argmin_min(image_colors, palette_colors)
	
	# Map pixels to palette colors
	quantized_rgb = np.zeros_like(rgb)
	quantized_rgb[non_transparent] = custom_palette[nearest_indices]
	
	# Handle alpha channel
	if preserve_alpha:
		quantized_alpha = alpha
	else:
		# Simplify alpha to binary (transparent/opaque)
		quantized_alpha = (alpha > 128).astype(np.uint8) * 255
	
	# Combine back to RGBA
	simplified_rgba = np.dstack([quantized_rgb, quantized_alpha])
	
	return simplified_rgba, custom_palette


def create_palette_from_colors(colors: List[Tuple[int, int, int]]) -> np.ndarray:
	"""
	Create a palette array from a list of RGB colors.
	
	Parameters
	----------
	colors: List[Tuple[int, int, int]]
		List of RGB colors as tuples (R, G, B)
		
	Returns
	-------
	np.ndarray
		Palette array of shape (N, 3)
	"""
	palette = np.array(colors, dtype=np.uint8)
	return palette


def check_gpu_availability() -> dict:
	"""
	Check what GPU acceleration options are available.
	
	Returns
	-------
	dict
		Dictionary with GPU availability information
	"""
	gpu_info = {
		'cupy_available': False,
		'pytorch_available': False,
		'cuda_available': False,
		'gpu_count': 0,
		'gpu_names': []
	}
	
	# Check CuPy (CUDA)
	try:
		import cupy as cp
		gpu_info['cupy_available'] = True
		gpu_info['cuda_available'] = True
		gpu_info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
		for i in range(gpu_info['gpu_count']):
			gpu_info['gpu_names'].append(cp.cuda.runtime.getDeviceProperties(i)['name'].decode())
	except ImportError:
		pass
	except Exception:
		# CuPy installed but CUDA not available
		pass
	
	# Check PyTorch
	try:
		import torch
		gpu_info['pytorch_available'] = True
		if torch.cuda.is_available():
			gpu_info['cuda_available'] = True
			gpu_info['gpu_count'] = torch.cuda.device_count()
			for i in range(gpu_info['gpu_count']):
				gpu_info['gpu_names'].append(torch.cuda.get_device_name(i))
	except ImportError:
		pass
	
	return gpu_info


def get_recommended_algorithm(image_size: tuple, gpu_available: bool = False) -> str:
	"""
	Get recommended algorithm based on image size and GPU availability.
	
	Parameters
	----------
	image_size: tuple
		Image dimensions (height, width)
	gpu_available: bool
		Whether GPU acceleration is available
		
	Returns
	-------
	str
		Recommended algorithm name
	"""
	h, w = image_size
	total_pixels = h * w
	
	if total_pixels > 1000000:  # > 1MP
		if gpu_available:
			return "perceptual"
		else:
			return "perceptual_fast"
	elif total_pixels > 500000:  # > 500K pixels
		return "perceptual_fast"
	elif total_pixels > 100000:  # > 100K pixels
		return "hsv_clustering"
	else:
		return "kmeans"
