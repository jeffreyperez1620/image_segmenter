#!/usr/bin/env python3
"""
Background worker thread for tendril cleanup operations.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PySide6.QtCore import QThread, Signal

# Add the app directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.tendril_trimming import trim_tendrils


class TendrilWorker(QThread):
    """Worker thread for running tendril cleanup algorithms."""
    
    progress_updated = Signal(int, int, str)  # current, total, message
    cleanup_completed = Signal(np.ndarray, int, str)  # result, iterations_used, status_message
    cleanup_failed = Signal(str)  # error_message
    
    def __init__(self, input_image, threshold, max_iterations):
        super().__init__()
        self.input_image = input_image
        self.threshold = threshold
        self.max_iterations = max_iterations
        
    def run(self):
        """Run the tendril cleanup algorithm."""
        try:
            self.progress_updated.emit(0, 100, "Starting tendril cleanup...")
            
            # Ensure input has alpha=255 for non-transparent pixels
            input_image = self._normalize_alpha_channel(self.input_image)
            
            # Run tendril cleanup with progress callback
            def progress_callback(current: int, total: int, message: str) -> None:
                # Convert to percentage
                progress = int((current / total) * 100) if total > 0 else 0
                self.progress_updated.emit(progress, 100, message)
            
            cleaned_output, iterations_used, status_message = trim_tendrils(
                input_image,
                self.threshold,
                self.max_iterations,
                progress_callback=progress_callback
            )
            
            if cleaned_output is not None:
                self.cleanup_completed.emit(cleaned_output, iterations_used, status_message)
            else:
                self.cleanup_failed.emit("Tendril cleanup failed - no output generated")
                
        except Exception as e:
            self.cleanup_failed.emit(f"Tendril cleanup error: {str(e)}")
    
    def _normalize_alpha_channel(self, rgba: np.ndarray) -> np.ndarray:
        """Ensure alpha=255 for non-transparent pixels."""
        result = rgba.copy()
        alpha = result[:, :, 3]
        
        # Set alpha to 255 for all non-transparent pixels
        non_transparent = alpha > 0
        result[non_transparent, 3] = 255
        
        return result
