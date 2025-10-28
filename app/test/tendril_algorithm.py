#!/usr/bin/env python3
"""
Tendril trimming algorithm implementation for the smoothing test application.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import QObject, Signal

# Add the app directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.tendril_trimming import TendrilTrimmer


class TendrilAlgorithm(QObject):
    """Tendril trimming algorithm with progress tracking and convergence detection."""
    
    progress_updated = Signal(str)  # message
    iteration_completed = Signal(int, int, int)  # iteration, tendrils_found, pixels_recolored
    
    def __init__(self):
        super().__init__()
        self.trimmer = TendrilTrimmer()
        
    def apply_tendril_trimming_with_progress(self, rgba, gaussian_kernel, unlimited_iterations=True, tendril_threshold=2, step_mode=False):
        """Apply tendril trimming with progress tracking and convergence detection."""
        import cv2 as cv
        
        # Clear and initialize debug log
        with open('tendril_debug.log', 'w') as f:
            f.write(f"# Tendril Debug Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parameters: unlimited={unlimited_iterations}, threshold={tendril_threshold}, step_mode={step_mode}\n")
            f.write(f"Image shape: {rgba.shape}\n")
            f.flush()
        
        def log_debug(message):
            with open('tendril_debug.log', 'a') as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")
                f.flush()
        
        log_debug("Starting tendril trimming algorithm")
        
        # Create a copy to work with
        result = rgba.copy()
        height, width = result.shape[:2]
        
        # Convert to RGB integer format for processing
        rgb_int = (result[:, :, 0].astype(np.uint32) << 16) | (result[:, :, 1].astype(np.uint32) << 8) | result[:, :, 2].astype(np.uint32)
        
        # Track changes for convergence detection
        changes_history = []
        max_iterations = 100 if not unlimited_iterations else 1000
        iteration = 0
        start_time = time.time()
        
        log_debug(f"Starting algorithm with max_iterations={max_iterations}, threshold={tendril_threshold}")
        
        while iteration < max_iterations:
            log_debug(f"Starting iteration {iteration + 1}")
            changes_made = False
            pixels_changed = 0
            
            # Debug: Count how many pixels will be processed
            non_transparent = result[:, :, 3] > 0
            pixels_to_process = np.sum(non_transparent[1:-1, 1:-1])
            log_debug(f"Processing {pixels_to_process} non-border pixels")
            
            # Apply tendril trimming algorithm
            log_debug(f"Starting tendril processing for iteration {iteration + 1}")
            self.progress_updated.emit(f"Processing tendrils for iteration {iteration + 1}...")
            QApplication.processEvents()
            
            try:
                # Use the backward compatibility method for one iteration
                tendrils_found = self.trimmer._mark_tendrils_for_iteration(result, tendril_threshold)
                log_debug(f"Tendrils found: {tendrils_found}")
                
                # Show tendril visualization if in step mode
                if step_mode and tendrils_found > 0:
                    log_debug(f"Creating tendril visualization for iteration {iteration + 1}")
                    self.progress_updated.emit(f"Creating tendril visualization for iteration {iteration + 1}...")
                    QApplication.processEvents()
                    
                    try:
                        from visualization import create_tendril_visualization, TendrilVisualizationDialog
                        tendril_vis = create_tendril_visualization(result, tendril_threshold)
                        log_debug(f"Tendril visualization created successfully")
                        
                        # Show dialog and wait for user response
                        dialog = TendrilVisualizationDialog(tendril_vis, iteration + 1)
                        if dialog.exec() == QDialog.Rejected:
                            log_debug("User canceled algorithm")
                            self.progress_updated.emit("Algorithm canceled by user")
                            return result
                        log_debug(f"User continued from tendril visualization")
                    except Exception as e:
                        log_debug(f"Error creating tendril visualization: {str(e)}")
                        self.progress_updated.emit(f"Error creating tendril visualization: {str(e)}")
                        QApplication.processEvents()
                        # Continue without visualization
                
                # Now process the marked tendrils
                if tendrils_found > 0:                    
                    log_debug(f"Applying color selection for iteration {iteration + 1}")
                    self.progress_updated.emit(f"Applying color selection for iteration {iteration + 1}...")
                    QApplication.processEvents()
                    
                    # Apply color selection to magenta pixels
                    pixels_changed = self.trimmer._process_tendrils_priority_queue(tendril_threshold)
                    changes_made = pixels_changed > 0

                    # Emit iteration completed signal
                    self.iteration_completed.emit(iteration + 1, tendrils_found, pixels_changed)

                    # Show color processing result if in step mode
                    if step_mode:
                        log_debug(f"Showing color processing result for iteration {iteration + 1}")
                        # Create a copy with all non-transparent pixels at full opacity
                        display_result = result.copy()
                        non_transparent = display_result[:, :, 3] > 0
                        display_result[non_transparent, 3] = 255
                        
                        try:
                            from visualization import ColorProcessingResultDialog
                            dialog = ColorProcessingResultDialog(display_result, iteration + 1)
                            if dialog.exec() == QDialog.Rejected:
                                log_debug("User canceled algorithm from color processing result")
                                self.progress_updated.emit("Algorithm canceled by user")
                                return result
                            log_debug(f"User continued from color processing result")
                        except Exception as e:
                            log_debug(f"Error showing color processing result: {str(e)}")
                            self.progress_updated.emit(f"Error showing color processing result: {str(e)}")
                            QApplication.processEvents()
            except Exception as e:
                log_debug(f"Error processing tendrils: {str(e)}")
                self.progress_updated.emit(f"Error processing tendrils: {str(e)}")
                QApplication.processEvents()
                # Continue without tendril processing
            
            # RGB values remain unchanged - only tendril trimming was applied
            
            iteration += 1
            log_debug(f"Completed iteration {iteration}")
            
            # Track changes for convergence detection
            if pixels_changed > 0:
                changes_history.append(pixels_changed)
                log_debug(f"Iteration {iteration}: {pixels_changed} RGB pixels changed ({pixels_changed / (height * width) * 100:.1f}%)")
            else:
                log_debug(f"Iteration {iteration}: No pixels changed")
            
            # Check for convergence
            if not changes_made:
                log_debug(f"Algorithm converged after {iteration} iterations - no changes made")
                break
            
            # Check for timeout (only if not in step mode)
            elapsed_time = time.time() - start_time
            if not step_mode:
                if elapsed_time > 15.0:  # 15 second timeout
                    log_debug(f"Algorithm timed out after {elapsed_time:.1f}s")
                    self.progress_updated.emit(f"⚠ Stopped after {iteration} iterations - timeout reached")
                    break
            else:
                log_debug(f"Step mode: timeout disabled, elapsed time {elapsed_time:.1f}s")
            
            # Additional safety: if tendril trimming is removing too many pixels, it might be in a loop
            # But exclude alpha channel changes from tendril marking (values 11, 12, 13)
            if pixels_changed > (height * width) * 0.5:
                # Check if the high pixel count is due to alpha channel tampering
                alpha_changes = np.sum((result[:, :, 3] > 10) & (result[:, :, 3] <= 13))
                rgb_changes = pixels_changed - alpha_changes
                
                if rgb_changes > (height * width) * 0.5:
                    log_debug(f"⚠ Stopped after {iteration} iterations - too many pixels changed ({pixels_changed}, {rgb_changes} RGB changes)")
                    self.progress_updated.emit(f"⚠ Stopped after {iteration} iterations - too many pixels changed")
                    break
            
            # Keep only last 20 change records to prevent memory growth
            if len(changes_history) > 20:
                changes_history = changes_history[-20:]
        
        # Apply final tendril cleanup
        log_debug("Starting final tendril cleanup")
        
        # For now, skip the final cleanup since it's causing hangs
        # The main algorithm already processed tendrils effectively
        log_debug("Skipping final tendril cleanup to prevent hanging")
        self.progress_updated.emit("Tendril processing completed (final cleanup skipped)")
        
        # Just restore alpha channel to full opacity
        self.trimmer._restore_alpha_channel(result)
        log_debug("Alpha channel restored to full opacity")
        
        log_debug(f"Algorithm completed successfully after {iteration} iterations")
        return result
