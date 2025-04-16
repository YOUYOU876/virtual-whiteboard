# Virtual Whiteboard Using Computer Vision

This project creates a virtual whiteboard experience using a webcam and computer vision techniques. There are three implementations provided with increasing levels of sophistication:

1. **Basic Virtual Whiteboard**: Color-based pen detection with simple drawing capabilities
2. **Enhanced Virtual Whiteboard**: Improved UI, multiple colors, eraser mode, and improved tracking
3. **Advanced ML Virtual Whiteboard**: Uses MediaPipe for hand tracking and supports both hand and color-based detection modes

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- MediaPipe (for the ML version)

Install the required packages:

```bash
pip install -r requirements.txt
```

## Features

- **Webcam Access**: Captures live video from your camera
- **Pen Detection**:
  - Color-based thresholding (all versions)
  - MediaPipe hand tracking (ML version)
- **Drawing Capabilities**:
  - Virtual canvas overlay on camera feed
  - Multiple color options
  - Adjustable pen thickness
  - Eraser mode
- **Interactive Controls**:
  - Keyboard shortcuts for all functions
  - Gesture recognition (ML version)
  - Color calibration for object tracking
- **Save and Export**: Save your drawings as image files

## Usage

### Basic Version

```bash
python virtual_whiteboard.py
```

### Enhanced Version

```bash
python virtual_whiteboard_enhanced.py
```

### ML Version (Recommended)

```bash
python virtual_whiteboard_ml.py
```

## Controls

| Key | Function                           |
| --- | ---------------------------------- |
| c   | Clear the canvas                   |
| s   | Save the current drawing           |
| d   | Toggle drawing mode on/off         |
| e   | Toggle eraser mode                 |
| +/- | Increase/decrease pen/eraser size  |
| 1-6 | Change drawing colors              |
| m   | Switch detection mode (ML version) |
| h   | Toggle help display (ML version)   |
| q   | Quit the application               |

## Calibration

When starting the application, you'll be asked if you want to calibrate the pen color detection:

1. Choose 'y' to enter calibration mode
2. Hold your pen/object in the green box in the center of the screen
3. Press 'c' to capture the color
4. Press 't' to test the calibration
5. Press 'a' to accept or any other key to continue testing
6. Press 'q' to exit calibration mode

## ML Version Gestures

The ML version supports hand gesture recognition:

- **Pinch gesture**: Pinch your thumb and middle finger together to clear the canvas

## Tips

- For best results with color detection, use a brightly colored object against a contrasting background
- Good lighting conditions will improve detection quality
- The ML version works best when your hand is clearly visible to the camera
- If tracking is unstable, try recalibrating or adjusting lighting

## Project Structure

- `virtual_whiteboard.py`: Basic implementation
- `virtual_whiteboard_enhanced.py`: Enhanced implementation with better UI and features
- `virtual_whiteboard_ml.py`: Advanced implementation with MediaPipe hand tracking
- `requirements.txt`: Required Python packages
- `whiteboard_captures/`: Directory where saved drawings are stored

## How It Works

1. **Pen Detection**:

   - **Color-based**: Converts frame to HSV color space, applies color thresholding, finds contours, and identifies the pen tip
   - **Hand tracking**: Uses MediaPipe to detect hand landmarks and tracks the index fingertip

2. **Drawing Mechanism**:

   - Tracks pen position between frames
   - Draws lines between consecutive points
   - Applies smoothing to reduce jitter

3. **UI Rendering**:
   - Overlays the virtual canvas on the camera feed
   - Displays control instructions and color palette
   - Shows visual feedback for the detected pen/finger

## Extending the Project

This project is designed to be modular and extensible. Here are some ideas for enhancements:

- Add more drawing tools (shapes, fill, text)
- Implement undo/redo functionality
- Add network capabilities for collaborative whiteboarding
- Create custom color picker
- Add image import/export features
- Implement different brush styles
- Create an eraser that only erases specific colors

## Troubleshooting

- **No camera access**: Ensure your webcam is properly connected and not in use by another application
- **Poor detection**: Try recalibrating the color detection or improving lighting conditions
- **MediaPipe errors**: Ensure you have the correct version installed (see requirements.txt)
- **Performance issues**: Lower the camera resolution in the code for better performance
