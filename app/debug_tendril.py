#!/usr/bin/env python3
"""
Simple debug script to test the tendril trimming algorithm
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processing.tendril_trimming import TendrilTrimmer

def create_test_image():
    """Create a simple test image with some thin features"""
    # Create a 100x100 image with some thin lines
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    
    # Add a thick rectangle (should not be a tendril)
    rgba[20:30, 20:80, :3] = [255, 0, 0]  # Red rectangle
    rgba[20:30, 20:80, 3] = 255  # Full alpha
    
    # Add a thin horizontal line (should be a tendril)
    rgba[50, 20:80, :3] = [0, 255, 0]  # Green line
    rgba[50, 20:80, 3] = 255  # Full alpha
    
    # Add a thin vertical line (should be a tendril)
    rgba[20:80, 70, :3] = [0, 0, 255]  # Blue line
    rgba[20:80, 70, 3] = 255  # Full alpha
    
    return rgba

def main():
    print("Creating test image...")
    test_image = create_test_image()
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Non-transparent pixels: {np.sum(test_image[:, :, 3] > 0)}")
    
    print("\nTesting tendril trimming with threshold=2...")
    trimmer = TendrilTrimmer()
    
    try:
        result, iterations, message = trimmer.trim_tendrils(test_image, threshold=2, max_iterations=5)
        print(f"Result: {message}")
        print(f"Iterations used: {iterations}")
        print(f"Result shape: {result.shape}")
        print(f"Non-transparent pixels in result: {np.sum(result[:, :, 3] > 0)}")
        
        # Check if debug log was created
        if os.path.exists("tendril_debug.log"):
            print("\nDebug log created! Contents:")
            with open("tendril_debug.log", "r") as f:
                print(f.read())
        else:
            print("\nNo debug log created.")
        
    except Exception as e:
        print(f"Error during tendril trimming: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
