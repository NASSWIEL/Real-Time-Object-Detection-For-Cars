# Main script for car detection in video
import cv2
from config import *
from model_loader import load_yolo_model, load_class_names, get_output_layers
from detector import detect_objects, process_detections, apply_non_max_suppression
from visualizer import draw_detections


def main():
    #  load yolo v3 model weights and config and classes
    print("Loading YOLO model...")
    yoloModel = load_yolo_model(MODEL_CONFIG, MODEL_WEIGHTS)
    classes = load_class_names(CLASS_NAMES)
    output_layers = get_output_layers(yoloModel)
    print("Model loaded successfully!")
    
    #  Open the video file
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH) # create an object to read video
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # Gets the frames per second (FPS) of the video.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Gets the width of the video frames in pixels.
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Gets the height of the video frames in pixels.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video : {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    #  Create video writer for output
    """
    FourCC code (short for Four Character Code) is a 4-character identifier used to specify a video codec
    It tells the video writer how to encode the frames so the output file can be played correctly.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    """
    out is an object you can use to save video frames to a file.

    OUTPUT_PATH : the filename where the video will be written.

    fourcc : the codec (compression format) used to encode the video.

    fps : the frame rate of the output video.

    (width, height) : the resolution of the output video.
    """
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    #  Process video frame by frame
    frame_count = 0
    
    while True:
        # Read next frame
        ret, frame = cap.read() # returns a boolean indicating success and the frame itself
        
        # Break if the end of the video is reached
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}...", end='\r')
        
        # Detect objects in the frame
        outputs, frame_width, frame_height = detect_objects(frame, yoloModel, output_layers, INPUT_SIZE)
        
        #  Process detections and filter for cars
        boxes, confidences, class_ids = process_detections(
            outputs, frame_width, frame_height, CONFIDENCE_THRESHOLD, CAR_CLASSES
        )
        
        # Apply non-maximum suppression to remove duplicate boxes
        indices = apply_non_max_suppression(boxes, confidences, NMS_THRESHOLD)

        # Draw bounding boxes and labels on the frame
        frame, vehicle_count = draw_detections(
            frame, boxes, confidences, class_ids, indices, classes, 
            BOX_COLOR, TEXT_COLOR, BOX_THICKNESS
        )
        
        # Write the processed frame to output video
        out.write(frame)
        
        # Show preview if enabled
        if SHOW_PREVIEW:
            cv2.imshow('Car Detection', frame)
            # Press 'q' to quit early
            if cv2.waitKey(1) == ord('q'): # wait 1 ms for key press, no key was pressed continue to the next frame
                break
    
    # Clean up resources
    cap.release()       # Release the VideoCapture object and free the video file or camera
    out.release()       # Release the VideoWriter object and finalize the output video file
    cv2.destroyAllWindows()  # Close all OpenCV windows opened by cv2.imshow()

    print(f"\nProcessing complete! Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
