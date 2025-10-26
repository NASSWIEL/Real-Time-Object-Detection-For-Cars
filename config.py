# Configuration file for car detection project

# Video settings
VIDEO_PATH = "vid.mp4"
OUTPUT_PATH = "output_detected.mp4"

# Model settings
MODEL_CONFIG = "yolov3.cfg"  # YOLO model configuration file
MODEL_WEIGHTS = "yolov3.weights"  # Pre-trained weights
CLASS_NAMES = "coco.names"  # File containing class names

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to detect an object (0-1)
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold to remove duplicate boxes
INPUT_SIZE = (416, 416)  # Input size for YOLO model

# Car-related class IDs from coco.names
CAR_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Display settings
SHOW_PREVIEW = True  # Show video while processing
BOX_COLOR = (0, 255, 0)  # Green color for bounding boxes (BGR format)
TEXT_COLOR = (255, 255, 255)  # White color for text
BOX_THICKNESS = 1  # Thickness of bounding box lines
