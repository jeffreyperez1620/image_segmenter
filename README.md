# Image Segmenter

A powerful desktop application for AI-powered background removal and intelligent color simplification of images.

## Features

### Background Removal
- **AI-Powered Removal**: Uses the `rembg` library for state-of-the-art background removal
- **Traditional Methods**: GrabCut algorithm for manual refinement
- **Portrait Matting**: Advanced closed-form matting for portrait refinement
- **Opacity Control**: Fine-tune transparency with adjustable opacity threshold
- **Interactive Brushing**: Manual foreground/background marking with brush tools

### Color Simplification
- **Multiple Algorithms**: Choose from 8 different color reduction methods
- **Perceptual Clustering**: Intelligently combine similar colors while preserving distinct regions
- **Memory Optimized**: Fast processing even for large images
- **GPU Acceleration**: Optional GPU support for faster processing
- **Color Statistics**: Detailed analysis of image colors and palette information

### User Interface
- **Modern Qt Interface**: Built with PySide6 for a native desktop experience
- **Tabbed Workflow**: Organized workflow from background removal to color simplification
- **Real-time Preview**: See changes instantly with live preview
- **Undo/Redo**: Full history support for all operations

## Color Simplification Algorithms

1. **K-means Clustering**: Best quality, uses machine learning clustering
2. **Median Cut**: Balanced approach, good for most images
3. **Octree Quantization**: Fast processing, efficient for large images
4. **Threshold/Posterize**: Simple quantization, good for artistic effects
5. **Perceptual Clustering**: Smart similar color grouping using LAB color space
6. **Perceptual Clustering (Fast)**: Memory-optimized version for large images
7. **Adaptive Distance**: Similar shades grouping with DBSCAN clustering
8. **HSV Clustering**: Color family preservation using HSV color space
9. **Adaptive**: Automatically selects the best algorithm based on image characteristics

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows, macOS, or Linux

### Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd image-segmenter

# Install required packages
pip install -r requirements.txt
```

### Optional GPU Acceleration

For faster processing on large images, you can install GPU acceleration:

```bash
# For CUDA support (if you have an NVIDIA GPU)
pip install cupy-cuda12x>=12.0.0

# For broader GPU support
pip install torch>=2.0.0
```

## Usage

### Running the Application

```bash
python -m app.main
```

### Workflow

1. **Load Image**: Open an image file through the File menu
2. **Background Removal**:
   - Use AI removal for automatic background removal
   - Or use GrabCut with manual brushing for precise control
   - Apply portrait matting for refined edges
   - Adjust opacity threshold as needed
3. **Color Simplification**:
   - Switch to the Color Simplification tab
   - Choose your preferred algorithm
   - Set the number of colors
   - Click "Simplify Colors" to see the result
   - Apply the simplification when satisfied

### Tips

- **Large Images**: Use "Perceptual Clustering (Fast)" for images larger than 1MP
- **Similar Colors**: Use "Perceptual Clustering" or "HSV Clustering" to combine similar shades
- **Artistic Effects**: Try "Threshold/Posterize" for stylized results
- **Memory Issues**: If you encounter memory problems, try the "Fast" versions of algorithms

## Technical Details

### Architecture
- **GUI Framework**: PySide6 (Qt for Python)
- **Image Processing**: OpenCV, NumPy, scikit-image
- **AI Background Removal**: rembg library
- **Color Processing**: scikit-learn for clustering algorithms
- **Advanced Matting**: pymatting for closed-form matting

### Performance
- **Memory Optimization**: Color sampling and downsampling for large images
- **GPU Acceleration**: Optional CUDA support via CuPy or PyTorch
- **Efficient Algorithms**: Optimized clustering and color mapping

## Development

### Project Structure
```
image-segmenter/
├── app/
│   ├── main.py                 # Application entry point
│   ├── processing/             # Image processing modules
│   │   ├── grabcut.py         # GrabCut implementation
│   │   ├── rembg_infer.py     # AI background removal
│   │   ├── matting_refine.py  # Portrait matting
│   │   └── color_simplify.py  # Color simplification algorithms
│   ├── ui/                    # User interface components
│   │   ├── main_window.py     # Main application window
│   │   ├── image_view.py      # Image display and interaction
│   │   ├── bg_tools_panel.py  # Background removal tools
│   │   └── color_simplify_panel.py  # Color simplification UI
│   └── utils/                 # Utility functions
│       └── qt_image.py        # Qt image conversion utilities
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- **rembg**: For AI-powered background removal
- **pymatting**: For advanced image matting algorithms
- **scikit-learn**: For clustering and machine learning algorithms
- **OpenCV**: For computer vision and image processing
- **PySide6**: For the modern Qt-based user interface
