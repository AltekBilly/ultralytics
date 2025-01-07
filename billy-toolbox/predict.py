from ultralytics import YOLO
import os
import cv2
from pathlib import Path

home_path = os.path.expanduser("~")

folder = '/home/ubuntu/dataset/GPU_pins_dataset_LZ20241203/val/images'
name = 'wiwynn-GPU_pins-20241205-'

# Load a model
model = YOLO(f'/home/ubuntu/repo/ultralytics/runs/detect/{name}/weights/best.pt') # load a custom model


save      = True
save_txt  = False #True
exist_ok  = True
save_crop = True
max_det   = 500

results = model(folder, save=save, save_txt=save_txt, exist_ok=exist_ok, save_crop=save_crop, max_det=max_det, name=f'predict-{name}')


# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     list_cls  = boxes.cls.tolist()
#     list_conf = boxes.conf.tolist()
#     list_xyxy = boxes.xyxy.tolist()
    
#     for idx, cls_idx in enumerate(list_cls):
#         if cls_idx != 3: continue
#         print(list_conf[idx])
        
    
    
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk
