# Cat Tracking System

A system for tracking and identifying individual cats using computer vision and deep learning, designed for Raspberry Pi 5 with Hailo-A AI accelerator.

## Overview

This system uses two cameras (regular and NIR) to detect, track, and identify cats. It includes a web dashboard for remote monitoring and automatic data collection for continually improving the model.

## Features

- **Dual Camera Tracking**: Utilizes both visible light and NIR cameras for robust detection
- **Individual Cat Identification**: Distinguishes between different cats based on visual features
- **Web Dashboard**: Monitor system status and cat detections remotely
- **Scheduled Operation**: Run the system at specific times of day
- **Automatic Data Collection**: Build a growing dataset for model improvement
- **Hardware Acceleration**: Support for Hailo-A AI HAT on Raspberry Pi 5

## Architecture

The system follows a modular architecture for maintainability:

```
├── config/                  # Configuration files
├── models/                  # Trained models directory
├── data/                    # Training/testing data
├── src/
│   ├── acquisition/         # Camera interface
│   ├── detection/           # Object detection (YOLO)
│   ├── tracking/            # Object tracking algorithms
│   ├── identification/      # Cat identification logic
│   ├── auto_annotation/     # Automatic data collection
│   ├── dashboard/           # Web interface
│   ├── scheduler/           # Time-based scheduling
│   └── utils/               # Utility functions
└── main.py                  # Application entry point
```

## Setup

### Prerequisites

- Raspberry Pi 5
- Hailo-A AI accelerator HAT
- Two cameras (visible and NIR)
- Python 3.9+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cat-tracking-system.git
   cd cat-tracking-system
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the system:
   Edit the YAML configuration files in the `config/` directory.

### Running with Stuffed Animals (Prototype)

For initial testing, you can use stuffed animals instead of real cats:

1. Set `use_dummy: false` in `config/cameras.yaml` to use real cameras (or keep `true` for simulated cameras)
2. Run the system:
   ```bash
   python main.py
   ```
3. Access the dashboard at `http://[raspberry-pi-ip]:8080`

## Configuration

The system uses YAML configuration files:

- `system.yaml`: Main system configuration
- `cameras.yaml`: Camera settings and synchronization
- `detection.yaml`: YOLO model and detection parameters
- `identification.yaml`: Cat identification parameters

## Development Workflow

1. Test with dummy cameras and stuffed animals
2. Collect initial training data
3. Retrain the model with your specific cats
4. Deploy for real cat tracking

## Using Transfer Learning

The system uses YOLOv8 with transfer learning for both detection and identification:

1. Start with pre-trained YOLOv8 model
2. Fine-tune on your collected cat images
3. Gradually improve the model as more data is collected

## Extending the System

- Add more cameras by extending the `CameraManager` class
- Implement different tracking algorithms in the `tracking` module
- Add new features to the web dashboard

## License

MIT

## Acknowledgments

- YOLOv8 by Ultralytics
- Hailo-A team for hardware acceleration