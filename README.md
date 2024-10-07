# Install
```shell
# Clone the ultralytics repository
git clone http://192.168.215.101:3000/BillyHsueh/ultralytics.git

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

# Train
```shell
cd ultralytics/
python billy_toolbox/train_landmark.py
```
train_landmark.py :
```python
...
date = "YYYYMMDD"

model_list = [
    ...,
    # 21 stride 32 - merl_rav - 20kpts
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-merl_rav-20p.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark_train_cfg.yaml", 
        "data": "altek-FacialLandmark-merl_rav-20p.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-merl_rav-20p-" + date + "-",
    },
    ...
]

model_idx = 21
...
# Load a model
model_path = '' # pre-train model path
...
```

# QAT
``` shell
cd ultralytics/
python billy_toolbox/train_landmark.py
```
train_landmark.py :
```python
...
date = "YYYYMMDD"

model_list = [
    ...,
    # 22 stride 32 - merl_rav - 20kpts -qat
    {
        "model": "yolov8n-Altek_Landmark-altek_FacailLandmark-merl_rav-20p.yaml", 
        "cfg": "Altek_Landmark-altek_FacialLandmark_qat_train_cfg.yaml", 
        "data": "altek-FacialLandmark-merl_rav-20p.yaml",
        "task": "altek_landmark",
        "name": "Altek_Landmark-FacialLandmark-merl_rav-20p-" + date + "-qat-",
    },
    ...
]

model_idx = 22
...
# Load a model
model_path = './runs/altek_landmark/[name]/weights/best.pt' # float pre-train model path
...
```

# Predict
```shell
cd ultralytics/
python billy_toolbox/predict_landmark.py
```
predict_landmark.py
```python
...
# Load a model
name = 'Altek_Landmark-FacialLandmark-3D_MODEL_IR_DATA-20240920-3' # model name
model_path = f'./runs/altek_landmark/{name}/weights/best.pt'
model = YOLO(model_path)  # load a custom model

# Define path to directory containing images and videos for inference
source = f'{home_path}/dataset/3D_MODEL_IR_DATA/images/val' # images path
name = f'preidct_{name}'
save_dir = f'./runs/altek_landmark/{name}'
...
```

# Label format
```yaml
[cls_index] [bbox_center_x] [bbox_center_y] [bbox_w] [bbox_h] [point_0_x] [point_0_y] [point_0_score] ... [point_i_x] [point_i_y] [point_i_score]
└────────────────────────BBOX───────────────────────────────┘ └────────────────────────────────────landmark─────────────────────────────────────┘
point_i_score = 2.0 # visible
point_i_score = 0.0 # invisible
```

# Training process
``` mermaid
graph TB
A[Train-billy_toolbox/train_landmark.py] 
-- runs/altek_landmark/name/weights/best.pt --> 
B[QAT-billy_toolbox/train_landmark.py] 
-- runs/altek_landmark/name-qat/weights/best.pt --> 
C[TinyNeuralNetwork]
--> 
D(best.tflite)
```

# The latest version of model.tflite
>[Altek_Landmark-FacialLandmark-merl_rav-20p-20240830-qat-3-best.tflite](http://192.168.215.101:3000/ToyotaLiu/M1_alDMS_DDAW/src/branch/master/FLD/src/Altek_Landmark-FacialLandmark-merl_rav-20p-20240830-qat-3-best.tflite)

# Dataset path
2024/08/30 - latest version dataset, 20 keypoints
```
/data/AIDATA/raw_data/2F-DATASET/merl_rav_for_yolov8-pose-20kpts/
```

3D IR Model - perspective landmark
```
/data/AIDATA/raw_data/2F-DATASET/3D_MODEL_IR_DATA/
```
