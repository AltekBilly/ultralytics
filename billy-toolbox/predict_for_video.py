from ultralytics import YOLO
import os
import cv2
from pathlib import Path

home_path = os.path.expanduser("~")

root = '/home/ubuntu/dataset/WYMY_High_Value_Room_Video_20240816/video'
name = 'wiwynn-High_Value_Room-20241125-'
video_name = '20240816_162752_CBE3_B8A44F494ED7_M1171184-001$A-RDimm.mkv'
video_path = os.path.join(root, video_name)

# Load a model
model = YOLO(f'/home/ubuntu/repo/ultralytics/runs/detect/{name}/weights/best.pt') # load a custom model
model.conf = 0.5  # 設定信心閾值 (0.0 - 1.0)
model.iou = 0.8   # 設定 IoU 閾值 (0.0 - 1.0)


cap = cv2.VideoCapture(video_path)

# 設定影片輸出 (可選)
output_path = os.path.join(root, f"{Path(video_path).stem}_predict.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

cls_idx = [5]
frame_idx = 0
# 逐幀處理影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 模型進行物件偵測
    results = model(frame)

    # 繪製偵測結果
    annotated_frame = results[0].plot()

    cls_number = {}
    cls_names = {}
    for result in results:
        cls_names = result.names
        for key in cls_names:
            cls_number[key] = 0
        
        boxes = result.boxes     
        for box in boxes:
            cls = int(box.cls.item())
            if cls in cls_idx:
                cls_number[cls] += 1
            else:
                continue
    
    for key in cls_names:
        count = cls_number[key]
        
        if count <= 0: continue
        
        test = f'{cls_names[key]}: {count}'
        cv2.rectangle(annotated_frame, (0, 0), (450, 80), (0,0,0), -1)
        cv2.putText(annotated_frame, test, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    
    # 顯示或保存偵測結果
    # cv2.imwrite('test.png', annotated_frame)
    out.write(annotated_frame)

# 釋放資源
cap.release()
out.release()

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk
