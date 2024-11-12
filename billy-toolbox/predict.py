from ultralytics import YOLO
import os
import cv2
from pathlib import Path

home_path = os.path.expanduser("~")

root = '/home/ubuntu/dataset/WYMY_High_Value_Room_Video_20240816/video'
name = 'wiwynn-High_Value_Room-20241107-3'
video_name = '20240816_110454_91F8_B8A44F494ED7_001_00ICX_M001-CPU.mkv'
video_path = os.path.join(root, video_name)

# Load a model
model = YOLO(f'/home/ubuntu/repo/ultralytics/runs/detect/{name}/weights/best.pt') # load a custom model



cap = cv2.VideoCapture(video_path)

# 設定影片輸出 (可選)
output_path = os.path.join(root, f"{Path(video_path).stem}_predict.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 逐幀處理影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 模型進行物件偵測
    results = model(frame)

    # 繪製偵測結果
    annotated_frame = results[0].plot()

    # 顯示或保存偵測結果
    # cv2.imshow("YOLOv8 Detection", annotated_frame)
    out.write(annotated_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
