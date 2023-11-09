import os
import cv2
import numpy as np
import colorsys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Keypoints

from pose_estimation import PoseEstimator

def draw_facial_component(img, component, color, isClosed=True):
    polyline = np.int32(component)
    cv2.polylines(img, [polyline], isClosed, color, 1, cv2.LINE_AA)

def draw_facial_label(img, kps_points):
    COLOR = (0,255,255)
    #
    for i in range(3):
        cv2.circle(img, kps_points[i], 2, COLOR, -1)

    # eyeball
    COLOR_eyeball = (255,255,0)
    cv2.circle(img, kps_points[17], 2, COLOR_eyeball, -1) # left
    cv2.circle(img, kps_points[18], 2, COLOR_eyeball, -1) # right

    # nose
    draw_facial_component(img, kps_points[3:5], COLOR, False)

    # eye-left
    draw_facial_component(img, kps_points[5:9], COLOR)

    # eye-right
    draw_facial_component(img, kps_points[9:13], COLOR)

    # mouth
    draw_facial_component(img, kps_points[13:17], COLOR)

def get_files_with_extension(folder_path: str, extension: str) -> list:
    file_paths = []
    
    for root, directories, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths

def draw_bbox(img: np.ndarray, bboxes: Boxes, index: int, scale):
    center_x, center_y, w, h = bboxes.xywhn[index].tolist()

    w = int(w*scale[1])
    h = int(h*scale[0])
    x = int(center_x*scale[1]- w/2) 
    y = int(center_y*scale[0]- h/2) 
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return center_x, center_y, w, h

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

def letter_box(image):

    size = 640

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算最长边和最短边的比例
    aspect_ratio = size / max(height, width)

    # 计算缩放后的高度和宽度
    if height > width:
        new_height = size
        new_width = int(width * aspect_ratio)
    else:
        new_width = size
        new_height = int(height * aspect_ratio)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建一个填充后的黑色背景
    padded_image = np.zeros((size, size, 3), dtype=np.uint8)

    # 计算填充的位置
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    # 将缩放后的图像放置在填充后的图像上
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
    return padded_image

def custom_sort(item):
    return item[0][1], item[0][0]

def draw_head_pose(img, kps_points, pose_estimator):
    marks = np.array([kps_points[4],  # Nose tip
                      kps_points[1],  # Chin
                      kps_points[5],  # Left eye left corner
                      kps_points[11], # Right eye right corner
                      kps_points[13], # Left Mouth corner
                      kps_points[15]  # Right mouth corner
                     ], dtype="double")
    pose = pose_estimator.solve(marks)
    pose_estimator.visualize(img, pose, color=(0, 255, 0))

def ckeck_dir(save_dir):
    # 使用os.path.exists()检查目录是否存在
    if not os.path.exists(save_dir):
        # 如果目录不存在，使用os.mkdir()创建目录
        os.mkdir(save_dir)

def save_image(save_dir, save_name, image):
    ckeck_dir(save_dir)
    # save image
    cv2.imwrite(f'{save_dir}/{save_name}', image)

def save_label(save_dir, save_name, label):
    ckeck_dir(save_dir)
    boxes     = label.boxes.xywhn.tolist()
    keypoints = label.keypoints.xyn.tolist()

    str_label = '0'
    for idx, boxe in enumerate(boxes):
        # bbox to string
        str_label += f' {boxe[0]:06f} {boxe[1]:06f} {boxe[2]:06f} {boxe[3]:06f}'
        # keypoints to string
        for keypoint in keypoints[idx]:
            str_label += f' {keypoint[0]:06f}  {keypoint[1]:06f}'
        #
        str_label += '\n'

    with open(f'{save_dir}/{save_name}', 'w') as file:
        file.write(str_label)

def save_mouth_keypoints(mouth_keypoints, save_dir, save_name):
    for i, (_, keypoint) in enumerate(mouth_keypoints):
        str_label = ''
        for point in keypoint:
            str_label += f"{point[0]:.6f} {point[1]:.6f} "
        with open(f'{save_dir}/{save_name}_{i}.txt', 'w') as file:
            file.write(str_label)

target_points = np.array([[0, 0], [0.5, 1], [1, 0]], dtype=np.float32)

def main(model, video_path: str, threshold: float = 0.5):
    # Load video
    cap = cv2.VideoCapture(video_path)

    # set save path
    save_root, video_name = os.path.split(video_path)
    video_name, video_ext = os.path.splitext(video_name)
    save_dir = f'{save_root}/{video_name}'

    # PoseEstimator
    pose_estimator = PoseEstimator(640, 480)

    counter = 0

    while cap.isOpened():
        success, oframe = cap.read()  

        # Break the loop if the end of the video is reached
        if not success: break

        if counter%30 == 0:
            print("counter: ", counter)
        else:
            counter+=1
            continue
        
        frame = oframe.copy() #letter_box(oframe)
        size = frame.shape
        results = model(frame) 

        # # save label
        # save_label(save_dir, f'{video_name}_{counter:06d}.txt', results[0])
        # # save image
        # save_image(save_dir, f'{video_name}_{counter:06d}.png', oframe)
        # counter+=1
        # continue

        keypoints = results[0].keypoints  # Masks object
        boxes = results[0].boxes
        objs_conf = boxes.conf
        objs_kp_point = keypoints.xyn.tolist()  # x, y keypoints (normalized), (num_dets, num_kpts, 2/3)
        mouth_keypoints = []
        
        do_draw = False
        for obj_index, obj_conf in enumerate(objs_conf):
            if obj_conf < threshold: continue
            
            # Draw bbox of face
            center_x, center_y, _, _ = draw_bbox(frame, boxes, obj_index, frame.shape)
            do_draw = True

            # Draw facial label
            _kps_points = objs_kp_point[obj_index]
            kps_points = [(int(_kps_points[point_index][0]*size[1]), int(_kps_points[point_index][1]*size[0])) for point_index in range(len(_kps_points))]
            draw_facial_label(frame, kps_points)
            
            # # Normalize the key-points of the mouth
            # source_points = np.array(kps_points[:3], dtype=np.float32)
            # transformation_matrix = cv2.getAffineTransform(source_points, target_points)
            # other_points = np.array(kps_points[13:17], dtype=np.float32)
            # transformed_points = cv2.transform(other_points.reshape(1, -1, 2), transformation_matrix).reshape(-1, 2)
            # mouth_keypoints.append([(center_x, center_y), transformed_points])

            # Draw head pose
            draw_head_pose(frame, kps_points, pose_estimator)

        # save mouth keypoints
        # save_mouth_keypoints(sorted(mouth_keypoints, key=custom_sort), save_dir, f'{counter:06d}')

        counter+=1
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)
        # if not do_draw: cv2.waitKey(0)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF ==ord('q'):
            break        


def load_filename_from_dir(dir, KEEP_EXT=True) -> list: 
    list = []
    for filename in os.listdir(dir):
        # 检查文件是否是普通文件而不是目录
        if os.path.isfile(os.path.join(dir, filename)):
            if any(filename.endswith(extension) for extension in ['.avi', '.mp4']):
                # 分离文件名和文件扩展名
                name, ext = os.path.splitext(filename)
                # 将文件名添加到列表中
                list.append(Path(dir)/filename if KEEP_EXT else Path(dir)/name)
    return list

if __name__ == '__main__':
    # model_path = 'yolov8n-pose-whiteboard-data_aug-640x640-33000-20230713.pt'
    model_path = './runs/pose/facial-landmark-19/weights/best.pt'

    # Load model
    model = YOLO(model_path)

    video_path = 'C:/Users/BillyHsueh/dataset/YawDD dataset/Mirror/Female_mirror/1-FemaleNoGlasses-Normal.avi'
    main(model, video_path, threshold=0.25)

    # video_dir = 'C:/Users/BillyHsueh/dataset/YawDD dataset/Mirror/Male_mirror Avi Videos'
    # video_list = load_filename_from_dir(video_dir)
    # for video_path in video_list:
    #     main(model, str(video_path), threshold=0.25)
        