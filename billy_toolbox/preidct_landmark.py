from ultralytics import YOLO
import os
# import numpy as np
import cv2

home_path = os.path.expanduser("~")

# Load a model
name = 'Altek_Landmark-FacialLandmark-test-20240701-stride64-CIB-reg_max4-2'
model_path = f'./runs/altek_landmark/{name}/weights/best.pt'
# model_path = './runs/pose/AltekPose-altek_FacialLandmark-test-20240626-2/weights/best.pt'
model = YOLO(model_path)  # load a custom model

# Define path to directory containing images and videos for inference
source = f'{home_path}/dataset/FacialLandmark_for_yolov8-pose-20240628/images/val'
name = f'preidct_{name}'
save_dir = f'./runs/altek_landmark/{name}'

# Predict with the model
results = model(source, imgsz=256, conf=0.5, cfg='altek_FacialLandmark_cfg.yaml', save_txt=True, save=True, name=name)  # generator of Results objects

# metrics  = model.val(
#     data='altek-FacialLandmark.yaml', 
#     cfg='altek_FacialLandmark_cfg.yaml', 
#     # name="test-model", batch= 64
#     )

# print(metrics)

for result in results:
    kps = result.keypoints.xy.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    img = result.orig_img
    file_name = os.path.basename(result.path)
    save_path = os.path.join(save_dir, file_name)
    
    for kp in kps[0]:
        img = cv2.circle(img, kp.astype(int), 2, (0,255,0), -1)
    
    img = cv2.rectangle(img, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (0,0,255))
        
    cv2.imwrite(save_path, img)