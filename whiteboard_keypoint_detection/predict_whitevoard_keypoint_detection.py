import os
import cv2
import numpy

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import ultralytics
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.engine.results import Keypoints

COLOR = [(  0,   0, 255), (  0,  64, 255), (  0, 128, 255), (  0, 191, 255),
         (  0, 255, 255), (  0, 255, 191), (  0, 255,   0), (191, 255,   0),
         (255, 255,   0), (255, 191,   0), (255,   0,   0), (255,   0, 191),
         (255,   0, 255), (191,   0, 255), (128,   0, 255), ( 64,   0, 255)]

def get_files_with_extension(folder_path: str, extension: str) -> list:
    file_paths = []
    
    for root, directories, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths

def get_line(points: list, confs: list, threshold: float = 0.5) -> tuple[tuple[float, float], tuple[float, float]]:  
    if len(points) != 5 or len(confs) != 5: return None

    count = 0
    for num in confs:
        if num > threshold:
            count += 1
            if count >= 2: break
    
    if count < 2: return None

    point0 = None
    point1 = None

    if confs[0] > threshold: point0 = points[0]
    if confs[4] > threshold: point1 = points[4]

    if point0 is None:
        for i in range(len(confs)):
            if confs[i] > 0.5:
                point0 = points[i]
                break

    if point1 is None:
        for i in range(len(confs)-1, -1, -1):
            if confs[i] > 0.5:
                point1 = points[i]
                break
    
    return ((point0[0], point0[1]), (point1[0], point1[1]))

def get_intersection_point(line0_p0: tuple[float, float], line0_p1: tuple[float, float], line1_p0: tuple[float, float], line1_p1: tuple[float, float]) -> list[float, float]:
    a = line0_p0[0] * line0_p1[1] - line0_p0[1] * line0_p1[0]
    b = line1_p0[0] * line1_p1[1] - line1_p0[1] * line1_p1[0]
    cx1 = line1_p0[0] - line1_p1[0]
    cx2 = line0_p0[0] - line0_p1[0]
    cy1 = line1_p0[1] - line1_p1[1]
    cy2 = line0_p0[1] - line0_p1[1]
    c = cx2 * cy1 - cy2 * cx1

    if c == 0: return None

    x = (a * cx1 - cx2 * b) / c
    y = (a * cy1 - cy2 * b) / c

    return [x, y]

def get_corners(points: list, confs: list) -> list[list[float]]:
    corners = [None, None, None, None]

    if len(points) != 16 or len(confs) != 16: return corners
    
    threshold = 0.5
    if(confs[0]  >= threshold): corners[0] = points[0]
    if(confs[4]  >= threshold): corners[1] = points[4]
    if(confs[8]  >= threshold): corners[2] = points[8]
    if(confs[12] >= threshold): corners[3] = points[12]

    if not any(item is None for item in corners): return corners

    line = [get_line(points[0:5], confs[0:5]), 
            get_line(points[4:9], confs[4:9]), 
            get_line(points[8:13], confs[8:13]), 
            get_line(points[12:16] + points[:1], confs[12:16] + confs[:1])]

    for i in range(len(corners)):
        if corners[i] is not None: 
            continue
        elif line[i] and line[i-1] is not None:
            corners[i] = get_intersection_point(line[i][0], line[i][1], line[i-1][0], line[i-1][1])
        else:
            corners[i] = points[i*4]

    return corners

def draw_bbox(img: numpy.ndarray, bboxes: Boxes, index: int, scale):
    x, y, w, h = bboxes.xywhn[index].tolist()

    w = int(w*scale[1])
    h = int(h*scale[0])
    x = int(x*scale[1]- w/2) 
    y = int(y*scale[0]- h/2) 
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

def draw_corner(img: numpy.ndarray, corners: list):
    if any(item is None for item in corners): return corners

    h, w, _ = img.shape

    for i in range(len(corners)):
        point0 = corners[i-1]
        point1 = corners[i]
        point0 = (int(point0[0] * w), int(point0[1] * h))
        point1 = (int(point1[0] * w), int(point1[1] * h))
        cv2.line(img, point0, point1, (0,0,255), 1, cv2.LINE_AA)

def write_to_txt(path:str, bboxes:Boxes, keypoints: Keypoints, threshold: float):
    with open(path, 'w') as file:
        list_xywh      = bboxes.xywhn.tolist()
        list_conf      = bboxes.conf.tolist()
        list_kps       = keypoints.xyn.tolist()
        list_kps_conf  = keypoints.conf.tolist() if keypoints.conf is not None else [[]]
        
        for i in range(len(list_xywh)):
            if(list_conf[i]<threshold): continue
            x, y, w, h = list_xywh[i]
            content = f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

            for j, kp in enumerate(list_kps[i]):
                content += f" {kp[0]:.6f} {kp[1]:.6f} {list_kps_conf[i][j]*2:.6f}"
            file.write(content + "\n")

def main(model_path:str, folder_path: str, extension: str = '.png', threshold: float = 0.5):
    all_file_paths = get_files_with_extension(folder_path, extension)

    # Load a model
    model = YOLO(model_path)
    # model = YOLO('keypoint-l-best.pt')
    model.overrides['imgsz'] = 1024
    
    for file_path in all_file_paths:
        print("================================")
        print(file_path)
        results = model(file_path)
        keypoints = results[0].keypoints  # Masks object
        boxes = results[0].boxes
        objs_conf = boxes.conf
        # keypoints.xy  # x, y keypoints (pixels), (num_dets, num_kpts, 2/3), the last dimension can be 2 or 3, depends the model.
        objs_kp_point = keypoints.xyn.tolist()  # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
        objs_kp_conf = keypoints.conf.tolist() if keypoints.conf is not None else [[]] # confidence score(num_dets, num_kpts) of each keypoint if the last dimension is 3.
        # keypoints.data  # raw keypoints tensor, (num_dets, num_kpts, 2/3) 


        # filename, ext = os.path.splitext(file_path)
        # write_to_txt(filename+".txt", boxes, keypoints, threshold)
        # continue

        img = cv2.imread(file_path)
        row, col, deep = img.shape
        scale = 640 / max(row, col)
        img = cv2.resize(src=img, dsize=(0,0), fx=scale, fy=scale)

        size = img.shape

        for obj_index, obj_conf in enumerate(objs_conf):
            if obj_conf < threshold: continue

            print("obj_conf:", obj_conf)
            draw_bbox(img, boxes, obj_index, img.shape)
            # continue

            # 
            kps_points = objs_kp_point[obj_index]
            kps_conf = objs_kp_conf[obj_index]

            #
            corners = get_corners(kps_points, kps_conf)
            draw_corner(img, corners)
            
            # draw
            for point_index in range(len(kps_points)):
                point = (int(kps_points[point_index][0]*size[1]), int(kps_points[point_index][1]*size[0]))
                conf = kps_conf[point_index]

                if (conf > 0.5): 
                    cv2.circle(img, point, 3, COLOR[point_index], -1)
                else:
                    cv2.circle(img, point, 3, COLOR[point_index], 1)

        cv2.imshow('img', img)
        
        if cv2.waitKey() == 27: break

if __name__ == '__main__':
    # model_path = 'yolov8n-pose-whiteboard-data_aug-640x640-33000-20230713.pt'
    model_path = './runs/pose/train-keypoint-n-640x640-data_aug/weights/best.pt'
    folder_path = 'C:/Users/BillyHsueh/dataset/Whiteboard-keypoints/images/val/altek'
    main(model_path, folder_path, threshold=0.5)