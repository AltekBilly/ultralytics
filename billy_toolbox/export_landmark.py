import sys
sys.path.append("/home/BillyHsueh/repo/ultralytics/")

from ultralytics import YOLO
import os
import numpy as np
import cv2

home_path = os.path.expanduser("~")

# Load a model
# model_path = './runs/pose/train8/weights/best.pt'

Is_Altek_Landmark = False #True # 

# Load a model
# model_path = 'yolov8n-Altek_Landmark-altek_FacailLandmark.yaml' if Is_Altek_Landmark else 'yolov8.yaml'
name = 'Altek_Landmark-FacialLandmark-test-20240709-stride64-3'
model_path = f'./runs/altek_landmark/{name}/weights/best.pt'
model = YOLO(model_path)#.load(model_path)  # build from YAML and transfer weights #yolov8n-pose-altek_FacailLandmark.yaml

# Export the model
int8 = True # False
# model.export(imgsz=256, format='tflite', int8=int8)
# model.export(imgsz=256, format='onnx')
model.export(imgsz=256, format='torchscript')