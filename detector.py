# Detection logic for identifying cars in frames
import cv2
import numpy as np


def detect_objects(frame, yoloModel, output_layers, input_size):
    height, width = frame.shape[:2]  # frame shape returns (height, width, channels)  [:2] take the first two values
    # preprocessed input image for the yoloModelwork. like Resize the image and normalize pixel values, and swap color channels
    # dnn is the module that runs the neural yoloModelwork without using high-level frameworks like TensorFlow or PyTorch. It operates directly on the image data.
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    # Set the blob as input to the yoloModelwork
    yoloModel.setInput(blob)
    # Run forward pass to get detections from output layers
    # because we are in inference phase, we use yoloModel only with one forward pass
    outputs = yoloModel.forward(output_layers)
    return outputs, width, height


def process_detections(outputs, width, height, confidence_threshold, car_classes):
    # Lists to store detection information
    boxes = []  # Bounding box coordinates
    confidences = []  # Detection confidence scores
    class_ids = []  # Detected class IDs
    
    # Loop through each output layer
    for output in outputs:
        # Loop through each detection in the output
        for detection in output:
            # First 5 values: x, y, w, h, objectness score
            scores = detection[5:]  #take scores for each class
            class_id = np.argmax(scores)  # Get class with highest score
            confidence = scores[class_id]  # Get confidence for that class
            
            # Only keep detections above threshold and car-related classes
            if confidence > confidence_threshold and class_id in car_classes:
                # YOLO returns center x, y, width, height (normalized 0-1)
                # ex detection = [0.5, 0.5, 0.2, 0.2, obj_score, class1_score, class2_score,...]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Convert to top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Store the detection
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids


def apply_non_max_suppression(boxes, confidences, nms_threshold):
    # Remove overlapping boxes (keep only the best one for each object)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0, nms_threshold)
    return indices
