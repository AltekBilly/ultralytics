import sys
sys.path.append("/home/BillyHsueh/repo/ultralytics/")

from ultralytics import YOLO
import os
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"

home_path = os.path.expanduser("~")

Is_Altek_Landmark = True# False# 

# Set path
model_path = 'yolov8n-Altek_Landmark-altek_FacailLandmark.yaml' if Is_Altek_Landmark else 'yolov8n-AltekPose-altek_FacailLandmark.yaml'
cfg_path = 'Altek_Landmark-altek_FacialLandmark_train_cfg.yaml' if Is_Altek_Landmark else 'AltekPose-altek_FacialLandmark_train_cfg.yaml'

# Load a model
# model_path = './runs/pose/train8/weights/best.pt'

# Load a model
model = YOLO(model_path)#.load(model_path)  # build from YAML and transfer weights #yolov8n-pose-altek_FacailLandmark.yaml

# Train the model
results = model.train(data='altek-FacialLandmark.yaml', cfg=cfg_path)

# Export the model
# model.export(imgsz=256, format='torchscript')