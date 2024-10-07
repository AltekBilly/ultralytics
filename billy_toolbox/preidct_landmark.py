from ultralytics import YOLO
import os
import glob
import numpy as np
import cv2

home_path = os.path.expanduser("~")

# Load a model
name = 'Altek_Landmark-FacialLandmark-3D_MODEL_IR_DATA-20240920-3'
model_path = f'./runs/altek_landmark/{name}/weights/best.pt'
model = YOLO(model_path)  # load a custom model

# Define path to directory containing images and videos for inference
source = f'{home_path}/dataset/3D_MODEL_IR_DATA/images/val'
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
    kps_conf = result.keypoints.conf.cpu().numpy()
    
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    img = result.orig_img
    file_name = os.path.basename(result.path)
    save_path = os.path.join(save_dir, file_name)
    
    image_path = result.path
    label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
    gt = np.loadtxt(label_path)[5:]
    gt = gt.reshape(-1, 3)
    gt[:, :2] = gt[:, :2]*256
    
    kps = kps[0]
    kps_conf = kps_conf[0]
    
    pd_img = img.copy()
    gt_img = img.copy()
    
    for idx, kp in enumerate(kps):
        # predict
        color = (0,0,255) if kps_conf[idx] < 0.5 else (255,0,0)
        pd_img = cv2.circle(pd_img, kp.astype(int), 2, color, -1)
        
        # GT
        color = (0,0,255) if gt[idx][2] < 0.5 else (255,0,0)
        gt_img = cv2.circle(gt_img, (gt[idx][:2]).astype(int), 2, color, -1)
        
    # img = cv2.rectangle(img, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), (0,0,255))
        
    cv2.imwrite(save_path, np.hstack((pd_img, gt_img)))