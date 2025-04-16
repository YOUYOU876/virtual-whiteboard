# Virtual Whiteboard

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10.x-blue)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8%2B-orange)](https://developers.google.com/mediapipe)

A computer vision-powered virtual whiteboard that enables drawing in mid-air using gesture recognition and object tracking.

## Overview

Virtual Whiteboard transforms any webcam-equipped computer into an interactive drawing surface by detecting hand movements and a designated drawing object. The system tracks when your finger touches the object and renders strokes on a virtual canvas, creating a natural drawing experience without specialized hardware.

## Features

- **Dual-hand Operation**: Designated drawing and control hands with distinct functions
- **Object Recognition**: Calibration system detects any physical object as a drawing tool
- **Touch Detection**: Precise tracking of finger-to-object contact points
- **Gesture Controls**:
  - Index finger gesture for color selection
  - Thumb gesture for stroke width adjustment
- **Whiteboard Modes**: Toggle between camera overlay and full whiteboard views
- **Writing Recognition**: Optional auto-correction of handwritten characters
- **Export Options**: Save drawings as image files

## Requirements

- Python 3.10.x
- OpenCV 4.5+
- NumPy 1.20+
- MediaPipe 0.8+

## Installation

```bash
# Clone the repository
git clone https://github.com/barandev/virtual-whiteboard.git
cd virtual-whiteboard

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
python virtual_whiteboard.py
```

On first run, the system will guide you through:
1. Hand registration (drawing vs. control hand)
2. Object calibration for drawing tool detection

## Usage

### Hand Registration

The system requires identification of your drawing and control hands:
- **Drawing Hand**: Holds the object and draws with the index finger
- **Control Hand**: Controls color/size selection via gestures

### Object Calibration

Any object with a distinct color can be used as a drawing tool:
1. Position object in the calibration frame
2. System samples its color profile for reliable detection
3. Default drawing color automatically matches the object

### Drawing Controls

| Action | Description |
|--------|-------------|
| Touch object with index finger | Begin drawing |
| Release finger from object | Stop drawing |
| Raise index finger (control hand) | Activate color selector |
| Raise thumb (control hand) | Activate size selector |
| Move control hand left/right | Change selected value |

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| w | Toggle whiteboard mode |
| c | Clear canvas |
| s | Save drawing |
| e | Toggle eraser mode |
| r | Toggle writing recognition |
| t | Adjust touch threshold |
| b | Toggle debug display |
| h | Toggle help overlay |
| q | Quit application |
| 1-8 | Select predefined colors |
| 0 | Use calibrated object color |
| +/- | Manual size adjustment |

## Technical Details

### Detection System

The application employs two primary detection methods:
- **MediaPipe Hands**: Tracks hand landmarks and finger positions
- **HSV Color Thresholding**: Identifies the calibrated drawing object

### Gesture Recognition

Hand gestures are detected using landmark relationships:
- **Index Finger Selection**: Extended index finger with other fingers curled
- **Thumb Selection**: Extended thumb with other fingers curled

### Touch Detection Algorithm

Touch is registered when the distance between finger and object tips falls below a configurable threshold, with a stability filter to prevent jitter.

### Writing Recognition

The optional writing recognition system:
1. Tracks completed strokes
2. Normalizes stroke geometry
3. Compares against character templates
4. Renders clean characters when matches are found

## Project Structure

```
virtual-whiteboard/
├── virtual_whiteboard.py      # Main application
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
└── whiteboard_captures/       # Saved drawings
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/X-feature`)
3. Commit your changes (`git commit -m 'Add X feature'`)
4. Push to the branch (`git push origin feature/X-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV community for computer vision tools
- MediaPipe team for hand tracking solutions