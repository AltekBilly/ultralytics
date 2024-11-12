from ultralytics import YOLO
import os
import cv2
from pathlib import Path

home_path = os.path.expanduser("~")

root = '/home/ubuntu/dataset/WYMY_High_Value_Room_Video_20240816/video'
name = 'wiwynn-High_Value_Room-20241107-3'
video_name = '20240816_110537_5773_B8A44F494ED7_001_00ICX_M001-CPU.mkv'
video_path = os.path.join(root, video_name)

# Load a model
model = YOLO(f'/home/ubuntu/repo/ultralytics/runs/detect/{name}/weights/best.pt') # load a custom model



cap = cv2.VideoCapture(video_path)

# 設定影片輸出 (可選)
output_path = os.path.join(root, f"{Path(video_path).stem}_track.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

count_area = [500, 500, 500+750, 500+420]
cpu_num = set()
# 逐幀處理影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 模型進行物件偵測
    results = model.track(frame, persist=True)

    boxes = results[0].boxes
    
    if boxes is not None and boxes.id is not None:
        cls = boxes.cls.cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()
        boxes = boxes.xywh.cpu().tolist()

        # count
        for idx, box in enumerate(boxes):
            if cls[idx] != 1 or box[0] < count_area[0] or box[1] < count_area[1] or box[0] > count_area[2] or box[1] > count_area[2]:
                continue
            
            cpu_num.add(track_ids[idx])
            
    # 繪製偵測結果
    annotated_frame = results[0].plot(conf=False)
    
    # draw
    annotated_frame = cv2.rectangle(annotated_frame, (count_area[0], count_area[1]), (count_area[2], count_area[3]), (0,0,64), 3)
    annotated_frame = cv2.rectangle(annotated_frame, (0, 0), (240, 80), (0,0,0), -1)
    text = f'cpu: {len(cpu_num)}'
    annotated_frame = cv2.putText(annotated_frame, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    
    # 保存偵測結果
    # cv2.imwrite("demo.png", annotated_frame)
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
