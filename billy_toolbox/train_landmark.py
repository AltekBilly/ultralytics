import sys
sys.path.append("/home/BillyHsueh/repo/ultralytics/")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# os.environ["WORLD_SIZE"] = "1"

from ultralytics import YOLO
import numpy as np
import cv2

home_path = os.path.expanduser("~")

date = "20240712"

model_list = [
    # 0 pose
    {
        "model": "yolov8n-AltekPose-altek_FacailLandmark.yaml", 
        "cfg": "AltekPose-altek_FacialLandmark_train_cfg.yaml", 
        "task": "pose",
        "name": "AltekPose-altek_FacialLandmark-test-" + date + "-",
    },
    # 1 stride 32
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark_train_cfg.yaml", 
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-",
    },
    # 2 stride 64
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-",
    },
    # 3 stride 128
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride128.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride128_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride128-",
    },
     # 4 stride 16
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride16.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride16_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride16-",
    },
    # 5 stride 128 - 512
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64-512.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64-512_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-512-",
    },
    # 6 stride 32 - 512
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-512.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-512_train_cfg.yaml", 
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-512-",
    },
    # 7 stride 32 - 128
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-128.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark_train_cfg.yaml", 
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-128-",
    },
    # 8 stride 64 - 20240701
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64-0701.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64-20240701_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-20240701-",
    },
    # 9 stride 64 - C1
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64-C1.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-C1-",
    },
    # 10 stride 64 - Bottleneck
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64-Bottleneck.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-Bottleneck-",
    },
    # 11 stride 64 - ResNetBlock
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64-ResNetBlock.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-ResNetBlock-",
    },
    # 12 stride 64 - CIB
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64-CIB.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-CIB-",
    },
    # 13 stride 64 - C3CIB
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64-C3CIB.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-C3CIB-",
    },
    # 14 stride 64 - qat
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-stride64.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark-stride64_qat_train_cfg.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-stride64-qat-",
    },
    # 15 stride 32 - 20240715
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-20240715.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark_train_cfg.yaml", 
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-test-" + date + "-",
    },
]

model_idx = 15

# Set path
model_path = model_list[model_idx]["model"]
cfg_path = model_list[model_idx]["cfg"]
name = model_list[model_idx]["name"] # "test-qat-model" # 

# Load a model
# model_path = './runs/altek_landmark/Altek_Landmark-FacialLandmark-test-20240709-stride64-3/weights/best.pt'

# Set task
task = model_list[model_idx]["task"]

# Load a model
model = YOLO(model=model_path, task=task)#.load(model_path)  # build from YAML and transfer weights #yolov8n-pose-altek_FacailLandmark.yaml

# Train the model
results = model.train(
    data='altek-FacialLandmark.yaml', 
    cfg=cfg_path, 
    name=name, 
    # batch= 1000,
    # epochs=10
    )

# Export the model
# model.export(imgsz=256, format='torchscript')