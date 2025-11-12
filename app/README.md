# Image Segmenter & SVG Layout

A tool to simplify images, segment by color, arrange segments for minimal layout, and export SVG for laser engraving.

## Features

- **Background Removal**: Remove backgrounds using AI (rembg) or manual masking tools
- **Color Processing**: Simplify colors in images using various algorithms
- **Region Cleanup**: Merge small regions, trim tendrils, and manually adjust regions with flood fill and brush tools
- **Arrange Regions**: Organize regions by color, rotate and position them, and export to SVG for laser engraving

## Requirements

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd image-segmenter/app
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **Start the application**:
   ```bash
   python main.py
   ```

2. **Load an image**:
   - Click **File → Open Image** or use `Ctrl+O`
   - Supported formats: PNG, JPG, JPEG, BMP, WebP

### Workflow

The application follows a 4-step workflow:

#### Step 1: Background Removal
- Remove the background using AI (rembg) or manual tools
- Use include/exclude masks to fine-tune the background removal
- Crop the image if needed

#### Step 2: Color Processing
- Simplify colors using various algorithms (K-means, Mean Shift, etc.)
- Use the eyedropper tool to select specific colors
- Reduce the number of colors in the image

#### Step 3: Region Cleanup
- Automatically merge small regions below a threshold
- Trim tendrils (thin protrusions) from regions
- Manually adjust regions using:
  - **Flood Fill**: Fill connected regions with a selected color
  - **Brush Tool**: Paint colors with adjustable brush size
  - **Undo/Redo**: Revert or reapply changes

#### Step 4: Arrange Regions
- View regions grouped by color in a palette table
- Select a color to view and arrange its regions
- **Drag and drop** regions to reposition them
- **Rotate regions** by dragging the rotation handle (circle around region centroid)
- **Export to SVG**:
  - Configure output dimensions (Width, Height)
  - Set units (inches, cm, mm)
  - Add margins if needed
  - Enable smoothing for smoother contours
  - Export selected color regions or all regions

### Keyboard Shortcuts

- `Ctrl+O`: Open image
- `Ctrl+S`: Save working image
- `Ctrl+Q`: Quit application

### Tips

- **For best results**: Use images with clear color separation
- **Color Processing**: Works best with 32 or fewer colors
- **Region Cleanup**: Adjust the merge threshold based on your image size
- **SVG Export**: Enable smoothing for cleaner engraving paths

## Project Structure

```
app/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── model/                  # Application state management
├── processing/             # Core image processing algorithms
│   ├── arrange_regions.py  # Region extraction and arrangement
│   ├── color_simplify.py   # Color simplification algorithms
│   ├── region_cleanup.py   # Region merging and cleanup
│   ├── svg_export.py        # SVG export functionality
│   └── ...
├── ui/                     # User interface components
│   ├── main_window.py      # Main application window
│   ├── image_view.py       # Image display widget
│   ├── arrange_regions/    # Arrange Regions step UI
│   ├── background_removal/ # Background removal step UI
│   ├── color_processing/   # Color processing step UI
│   └── region_cleanup/     # Region cleanup step UI
└── utils/                  # Utility functions
```

## Dependencies

- **PySide6**: Qt framework for GUI
- **numpy**: Numerical operations
- **opencv-python**: Image processing
- **scipy**: Scientific computing (for spline interpolation)
- **scikit-image**: Image processing algorithms
- **scikit-learn**: Machine learning (for color clustering)
- **rembg**: AI background removal
- **pymatting**: Matting algorithms
- **shapely**: Geometric operations

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Make sure you're using the correct Python version (3.8+)

### Performance Issues
- Large images may take longer to process
- Consider resizing very large images before processing
- GPU acceleration is optional (see requirements.txt)

### SVG Export Issues
- Ensure output dimensions are valid numbers
- Check that margins don't exceed output dimensions
- Verify file path has write permissions

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here if applicable]

