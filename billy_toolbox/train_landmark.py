import sys
sys.path.append("/home/BillyHsueh/repo/ultralytics/")

from ultralytics import YOLO
import os
import numpy as np
import cv2

home_path = os.path.expanduser("~")

# Load a model
# model_path = './runs/pose/train8/weights/best.pt'

# Load a model
model = YOLO('yolov8n-AltekPose-altek_FacailLandmark.yaml')#.load(model_path)  # build from YAML and transfer weights #yolov8n-pose-altek_FacailLandmark.yaml

# Train the model
results = model.train(data='altek-FacialLandmark.yaml', cfg='AltekPose-altek_FacialLandmark_train_cfg.yaml')

# Export the model
# model.export(imgsz=256, format='torchscript')