import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

# 讀取CSV文件
csv_file = 'billy_toolbox/IR_val_20240604_heatmap_eye_visible.csv' #'../FacialLandmarkHeatmap/IR_val_20240319_heatmap.csv'
data = pd.read_csv(csv_file)


# 創建輸出目錄
home_path = os.path.expanduser("~")
output_image_dir = f'{home_path}/dataset/FacialLandmark_visible_for_yolov8-pose/images'
output_label_dir = f'{home_path}/dataset/FacialLandmark_visible_for_yolov8-pose/labels'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 目標大小
target_size = 256

# 遍歷每一行數據
for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing"):
    image_path = row['image']
    cx, cy, L = row['cx'], row['cy'], row['L']
    keypoints = row[4:].values

    # 打開圖像
    img = Image.open(image_path)
    
    # 計算裁剪區域
    expanded_L = L * 1.1
    left = cx - expanded_L / 2
    top = cy - expanded_L / 2
    right = cx + expanded_L / 2
    bottom = cy + expanded_L / 2

    # 裁剪圖像
    cropped_img = img.crop((left, top, right, bottom))

    # 調整裁剪後圖像的大小為256x256
    resized_img = cropped_img.resize((target_size, target_size))

    # 保存裁剪後並縮放的圖像
    image_name = os.path.basename(image_path)
    resized_image_path = os.path.join(output_image_dir, image_name)
    resized_img.save(resized_image_path)

    # 計算縮放比例
    scale = target_size / expanded_L

    # 計算新的中心點
    new_cx = 0.5
    new_cy = 0.5
    new_L = L*scale / target_size
    
    # 調整key-points座標
    adjusted_keypoints = []
    for i in range(0, len(keypoints), 3):
        x, y, occluded = keypoints[i], keypoints[i+1], keypoints[i+2]
        adjusted_x = (x - left) * scale
        adjusted_y = (y - top) * scale
        visible = (1 - occluded) * 2
        adjusted_keypoints.extend([adjusted_x, adjusted_y, visible])

    # 構建txt內容
    txt_content = f"0 {new_cx:.6f} {new_cy:.6f} {new_L:.6f} {new_L:.6f}"
    for i in range(0, len(adjusted_keypoints), 3):
        x, y, visible = adjusted_keypoints[i]/target_size, adjusted_keypoints[i+1]/target_size, adjusted_keypoints[i+2]
        txt_content += f" {x:.6f} {y:.6f} {visible:.6f}"
    txt_content = txt_content.strip()

    # 保存txt文件
    txt_file_path = os.path.join(output_label_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(txt_file_path, 'w') as f:
        f.write(txt_content)

print("處理完成！")