#!/usr/bin/env python3
"""
Background worker thread for smoothing operations.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThread, Signal

# Add the app directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.region_cleanup import smooth_region_boundaries


class SmoothingWorker(QThread):
    """Worker thread for running smoothing algorithms."""
    
    progress_updated = Signal(int, int, str)  # current, total, message
    algorithm_completed = Signal(str, np.ndarray, float)  # algorithm_name, result, time_taken
    
    def __init__(self, input_image, strength, preserve_colors):
        super().__init__()
        self.input_image = input_image
        self.strength = strength
        self.preserve_colors = preserve_colors
        
    def run(self):
        """Run the smoothing algorithms."""
        algorithms = [
            ("Original", lambda img: img.copy()),
            ("Region Cleanup", lambda img: smooth_region_boundaries(img, self.strength, self.preserve_colors))
        ]
        
        total = len(algorithms)
        
        for i, (algorithm, func) in enumerate(algorithms):
            try:
                self.progress_updated.emit(i, total, f"Applying {algorithm}...")
                
                start_time = time.time()
                result = func(self.input_image)
                end_time = time.time()
                time_taken = end_time - start_time
                self.algorithm_completed.emit(algorithm, result, time_taken)
            except Exception as e:
                print(f"Error applying {algorithm}: {e}")
                # Create a copy of input as fallback
                self.algorithm_completed.emit(algorithm, self.input_image.copy(), 0.0)
        
        self.progress_updated.emit(total, total, "All algorithms completed!")
