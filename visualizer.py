# Draw bounding boxes and labels on frames
import cv2


def draw_detections(frame, boxes, confidences, class_ids, indices, classes, box_color, text_color, thickness):
    # Counter for detected vehicles
    vehicle_count = 0
    
    # Loop through the indices of boxes that passed NMS
    if len(indices) > 0:
        for i in indices.flatten():
            # Get box coordinates
            x, y, w, h = boxes[i]
            
            # Get class name and confidence
            label = classes[class_ids[i]]
            confidence = confidences[i]
            
            # Draw rectangle around detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)
            
            # Create label text with class name and confidence
            text = f"{label}: {confidence:.2f}"
            
            # Draw label background (filled rectangle)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), box_color, -1)
            
            # Draw label text
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            vehicle_count += 1
    
    # Draw total count on the frame
    count_text = f"Vehicles Detected: {vehicle_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, vehicle_count
