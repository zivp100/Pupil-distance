# Pupil Distance Measurement Script

This script measures inter-pupillary distance with sub-millimeter accuracy using MediaPipe Face Mesh for iris detection and a credit card as a reference scale.

## Requirements

- Python 3.8+
- MediaPipe
- OpenCV
- NumPy

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you encounter permission issues, try:

```bash
pip install --user -r requirements.txt
```

**Note**: The script will automatically download the face landmarker model (face_landmarker.task) on first run. The model will be saved to `~/.mediapipe_models/` directory.

## Usage

### Basic Usage

Process a single image:

```bash
python pupil_distance_measurement.py path/to/image.jpg
```

### Specify Output Directory

```bash
python pupil_distance_measurement.py path/to/image.jpg -o output_directory
```

## Output Files

For each input image, the script generates:

1. **`{input_name}_pupils_marked.jpg`**: Visualization with pupils marked (red circles) and inter-pupillary distance labeled
2. **`{input_name}_card_detected.jpg`**: Visualization with credit card detection rectangle (blue) and width labeled
3. **`{input_name}_cropped_roi.jpg`**: Cropped region of interest (from nose to chin, full face width) where the credit card is searched
4. **`{input_name}_results.json`**: JSON file containing all measurements:
   - Pupil distance (pixels and millimeters)
   - Credit card width (pixels and millimeters)
   - Conversion factor (mm/pixel)
   - Pupil coordinates
   - Paths to output images

## How It Works

1. **Face and Iris Detection**: Uses MediaPipe Face Mesh with refined landmarks to detect iris landmarks
2. **Pupil Center Calculation**: Calculates pupil centers as centroids of iris landmark points
3. **Credit Card Detection**: Detects credit card below the nose using edge detection and contour analysis
4. **Distance Measurement**: Calculates inter-pupillary distance in pixels
5. **Scale Conversion**: Uses standard credit card width (85.60mm) to convert pixel measurements to millimeters
6. **Visualization**: Generates annotated images for verification

## Credit Card Standard

The script uses the ISO 7810 ID-1 standard credit card width of **85.60mm** for scale conversion.

## Error Handling

The script will raise errors if:
- Image cannot be loaded
- Face or iris is not detected
- Credit card is not detected

Make sure the input image shows:
- A clear frontal face view
- Visible eyes with open irises
- A credit card positioned below the nose
