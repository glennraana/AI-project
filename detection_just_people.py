# We update the code so we just see the person in the frame, not everything around. 
# This code take a picture of the frame so we can see that things work. Another thing that is great with 
# This program is that you can find the x and y values for where you want to put the polygon .

import supervision as sv
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
import matplotlib.pyplot as plt

model = YOLO('/Users/glenn/counting_people_in_zone/virtualspace/yolov8s.pt')

# extract video frame
generator = sv.get_video_frames_generator(source_path="/Users/glenn/counting_people_in_zone/virtualspace/testvideo2.mov")
iterator = iter(generator)
frame = next(iterator)

# detect
results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.class_id == 0]

# annotate
box_annotator = sv.BoundingBoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
frame = box_annotator.annotate(scene=frame, detections=detections)
frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

sv.plot_image(frame, (16, 16))