# Load and prepare the YOLO model for detection
import cv2
import numpy as np


def load_yolo_model(config_path, weights_path):
    # Load YOLO neural network from config and weights files
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use CPU for processing
    return net


def load_class_names(names_path):
    # Read all class names from the file (one per line)
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def get_output_layers(net):
    # Get the names of all layers in the network
    layer_names = net.getLayerNames()
    # Get only the output layer names (where detections come from)
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
