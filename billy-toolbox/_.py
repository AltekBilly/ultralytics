import os

# 定義資料夾的路徑
root_dir = '/home/ubuntu/dataset/WYMY_High_Value_Room_Video_20240816'

# 遍歷 'train/labels' 和 'val/labels' 資料夾
for folder in ['train/labels', 'val/labels']:
    labels_dir = os.path.join(root_dir, folder)
    
    # 遍歷所有子資料夾
    for subdir, _, files in os.walk(labels_dir):
        for file in files:
            if file.endswith('.txt') and file != 'classes.txt':
                file_path = os.path.join(subdir, file)
                
                # 讀取每個 .txt 文件中的內容
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # 存儲更新後的內容
                updated_lines = []
                
                for line in lines:
                    # 分割每一行以獲取 class_id 和其他 YOLO 座標
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    
                    # 檢查 class_id 並進行相應處理
                    if class_id == 5:
                        parts[0] = '4'  # 將 class_id 5 改為 4
                        updated_lines.append(' '.join(parts))
                    elif class_id != 2:
                        updated_lines.append(line.strip())  # 保留其他 class_id 的標籤
                
                # 將更新後的內容寫回 .txt 文件
                with open(file_path, 'w') as f:
                    for updated_line in updated_lines:
                        f.write(updated_line + '\n')
