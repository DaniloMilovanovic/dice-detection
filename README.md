# Dice Detection and Recognition System

A computer vision project that automatically detects dice in images, identifies their values, and calculates their sizes using Python and OpenCV.

## Overview

This system processes images of dice and performs the following tasks:
1. Loads and preprocesses images using HSV color space
2. Binarizes and cleans images using morphological operations
3. Detects dots on dice using Difference of Hessian (DoH) blob detection
4. Clusters dots into individual dice using DBSCAN algorithm
5. Calculates dice sizes based on dot distribution
6. Visualizes results with annotated dice values and boundaries

## Features

- **Automatic Dice Detection**: Identifies multiple dice in a single image
- **Value Recognition**: Counts dots to determine each die's value
- **Size Calculation**: Computes dice dimensions based on dot distribution
- **Visualization**: Displays detected dice with boundaries and annotations
- **Batch Processing**: Handles multiple images sequentially

## Algorithm Details

### Image Processing
- Conversion from RGB to HSV color space
- Saturation channel extraction for better contrast
- Binary thresholding and morphological operations (opening and erosion)

### Dot Detection
- Uses Difference of Hessian (DoH) blob detection
- Parameters: `min_sigma=5`, `max_sigma=15`, `threshold=0.05`

### Dice Clustering
- DBSCAN algorithm with `eps=35` and `min_samples=1`
- Groups dots into individual dice based on spatial proximity

### Size Calculation
- Determines dice size from dot distribution
- Uses bounding box of dots with padding
- Applies constraints (30-100 pixel range)

## Sample Output

The program generates visualizations showing:
- Original image with detected dots
- Dice boundaries based on calculated sizes
- Text annotations with dice values and dimensions
- Console output with processed results


## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Dependencies

Install required packages:

pip install matplotlib scikit-image scikit-learn numpy

## Usage

1. Place your dice images in the project directory named as `dices1.jpg`, `dices2.jpg`, etc.
2. Run the main script:

python dice_detection.py


3. The script will process each image and display results with:
   - Detected dots (red circles with black centers)
   - Dice boundaries (blue dashed circles)
   - Annotations showing dice values and sizes


## Limitations

- Requires clear contrast between dots and dice surface
- Works best with standard dice layouts and colors
- May have difficulty with overlapping dice or extreme lighting conditions

## Future Improvements

- Support for various dice colors and materials
- Handling of rotated or partially obscured dice
- 3D dice recognition and orientation detection
- Real-time video processing capabilities
