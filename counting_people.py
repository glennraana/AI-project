import numpy as np
import cv2
import supervision as sv
import torch
import os
import ultralytics
from setuptools import distutils
import torch, detectron2
import supervision as sv
from ultralytics import YOLO
model = YOLO('yolov8s.pt')



# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera. Use 1, 2, etc. if you have multiple cameras

# Define a polygon covering the right half of a 1080x720 frame
polygon = np.array([
    [540, 0],        # Top middle of the frame
    [1080, 0],       # Top right corner
    [1080, 720],     # Bottom right corner
    [540, 720],      # Bottom middle of the frame
])

# Since this is a live stream, we'll need to set the frame width and height manually
frame_width = 1080
frame_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

zone = sv.PolygonZone(polygon=polygon)

# Initiate annotators
box_annotator = sv.BoundingBoxAnnotator(thickness=4)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)

# Initialize variables to track person counts
person_count = 0
current_people_in_zone = set()

def process_frame(frame: np.ndarray, i: int) -> np.ndarray:
    global person_count, current_people_in_zone

    # Detect
    outputs = predictor(frame)
    detections = sv.Detections(
        xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
        confidence=outputs["instances"].scores.cpu().numpy(),
        class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int)
    )
    detections = detections[detections.class_id == 0]  # Filter to keep only persons (class_id == 0)

    # Get detected person IDs
    person_ids = set(range(len(detections.xyxy)))

    # Check if detections are within the zone
    zone_results = zone.trigger(detections=detections)

    # Update the count of people in the zone
    people_entering = person_ids - current_people_in_zone
    people_leaving = current_people_in_zone - person_ids

    # Add new people entering the zone
    current_people_in_zone.update(people_entering)

    # Remove people who have left the zone
    current_people_in_zone.difference_update(people_leaving)

    # Update the count based on current people in the zone
    person_count = len(current_people_in_zone)

    # Annotate
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = zone_annotator.annotate(scene=frame)

    # Display the count of persons detected in the zone using OpenCV
    label_text = f'Persons in Zone: {person_count}'
    position = (50, 50)  # Position to display the label on the video frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)  # White color in BGR
    thickness = 2
    cv2.putText(frame, label_text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    return frame

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 1080x720 if needed
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Process the frame
    processed_frame = process_frame(frame, i)

    # Display the processed frame
    cv2.imshow('Webcam Stream', processed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()