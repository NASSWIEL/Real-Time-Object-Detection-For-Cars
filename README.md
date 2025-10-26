# Real-Time Car Detection System

A production-ready vehicle detection system using YOLOv3 deep learning model to identify and track cars, motorcycles, buses, and trucks in video footage.

## Features

- **Real-time Detection**: Processes video frame-by-frame with bounding box visualization
- **Multi-vehicle Support**: Detects cars, motorcycles, buses, and trucks
- **Modular Architecture**: Clean, maintainable code structure
- **Configurable Parameters**: Easy threshold and setting adjustments
- **Live Preview**: See detections as they happen
- **Video Output**: Saves processed video with annotations

## Project Structure

```
Object Detection/
├── config.py              # Configuration and settings
├── model_loader.py        # YOLO model initialization
├── detector.py            # Detection algorithms
├── visualizer.py          # Visualization functions
├── main.py                # Main execution script
├── vid.mp4                # Input video
└── output_detected.mp4    # Output with detections
```

## Installation

1. Install dependencies:
```bash
pip install opencv-python numpy
```

2. Download YOLO model files and place in project directory:
   - `yolov3.cfg` - Already included in repository
   - `coco.names` - Already included in repository
   - `yolov3.weights` - **If you want the weights, you will find them in this link:** [Download YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights) (236 MB)

## Usage

Run the detection system:
```bash
python main.py
```

Press 'q' during preview to stop processing early.

## Configuration

Edit `config.py` to customize:

- `CONFIDENCE_THRESHOLD`: Detection sensitivity (default: 0.5)
- `NMS_THRESHOLD`: Overlap suppression (default: 0.4)
- `VIDEO_PATH`: Input video file path
- `BOX_COLOR`: Bounding box color (BGR format)
- `SHOW_PREVIEW`: Enable/disable live preview

## Technical Details

- **Model**: YOLOv3 (You Only Look Once v3)
- **Framework**: OpenCV DNN module
- **Input Size**: 416x416 pixels
- **Classes**: COCO dataset (80 classes, filtered for vehicles)
- **Processing**: CPU-based inference

## How It Works

1. **Model Loading**: Initializes YOLOv3 neural network with pre-trained weights
2. **Video Input**: Reads video frame by frame
3. **Preprocessing**: Converts frames to blob format for neural network
4. **Detection**: Runs forward pass through YOLO layers
5. **Post-processing**: Filters detections by confidence and applies NMS
6. **Visualization**: Draws bounding boxes and labels
7. **Output**: Saves annotated video

## Performance

- **Speed**: ~2-5 FPS on CPU (depends on hardware)
- **Accuracy**: ~50-80% detection rate for clear vehicle footage
- **Classes Detected**: Car (ID: 2), Motorcycle (ID: 3), Bus (ID: 5), Truck (ID: 7)

## Skills Demonstrated

- Computer Vision & Deep Learning
- OpenCV & Neural Networks
- Video Processing Pipelines
- Modular Python Architecture
- Object Detection Algorithms
- Real-time Data Processing

## Future Enhancements

- [ ] Vehicle tracking across frames
- [ ] Speed estimation
- [ ] License plate detection
- [ ] GPU acceleration support
- [ ] Multi-camera support
- [ ] Statistical reporting

## License

Open source - Free to use for educational and portfolio purposes.

## Author

[Your Name] - Computer Vision Engineer

---
*Built with Python, OpenCV, and YOLOv3*
