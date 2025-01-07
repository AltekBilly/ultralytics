from ultralytics import YOLO
import os

home_path = os.path.expanduser("~")

date = "20241205"


model_dic = {
    "wiwynn-RPN_pins-cfg": {
        "model": "yolov8s.yaml", 
        "cfg": "wiwynn-RPN_pins-cfg.yaml", 
        "data": "wiwynn-RPN_pins.yaml",
        "task": "detect",
        "name": "wiwynn-RPN_pins-" + date + "-",
    },
    
    "wiwynn-High_Value_Room": {
        "model": "yolov8s.yaml", 
        "cfg": "wiwynn-High_Value_Room-cfg.yaml", 
        "data": "wiwynn-High_Value_Room.yaml",
        "task": "detect",
        "name": "wiwynn-High_Value_Room-" + date + "-",
    },
    
    "wiwynn-GPU_pins": {
        "model": "yolov8s.yaml", 
        "cfg": "wiwynn-GPU_pins-cfg.yaml", 
        "data": "wiwynn-GPU_pins.yaml",
        "task": "detect",
        "name": "wiwynn-GPU_pins-" + date + "-",
    },
    
    "wiwynn-GPU_pins_LZ20241203": {
        "model": "yolov8s.yaml", 
        "cfg": "wiwynn-GPU_pins_LZ20241203-cfg.yaml", 
        "data": "wiwynn-GPU_pins_LZ20241203.yaml",
        "task": "detect",
        "name": "wiwynn-GPU_pins-" + date + "-",
    },
    
    "wiwynn-GPU_pins_Classification": {
        "model": "yolov8n-cls.yaml", 
        "cfg": "wiwynn-GPU_pins_Classification-cfg.yaml", 
        "data": "/home/ubuntu/dataset/GPU_pins_Classification_dataset",
        "task": "classify",
        "name": "wiwynn-GPU_pins_Classification-" + date + "-",
    },
    
    "wiwynn-GPU_pins_Classification_LZ20241203": {
        "model": "yolov8n-cls.yaml", 
        "cfg": "wiwynn-GPU_pins_Classification_LZ20241203-cfg.yaml", 
        "data": "/home/ubuntu/dataset/GPU_pins_Classification_dataset_LZ20241203",
        "task": "classify",
        "name": "wiwynn-GPU_pins_Classification-" + date + "-",
    },
    
    "wiwynn-High_Value_Room-obb": {
        "model": "yolo11s-obb.yaml", 
        "cfg": "wiwynn-High_Value_Room-obb-cfg.yaml", 
        "data": "wiwynn-High_Value_Room-obb.yaml",
        "task": "obb",
        "name": "wiwynn-High_Value_Room-obb-" + date + "-",
    },
}

model_key = "wiwynn-GPU_pins_Classification_LZ20241203"

# Set path
model_path = model_dic[model_key]["model"]
cfg_path = model_dic[model_key]["cfg"]
data = model_dic[model_key]["data"]
name = model_dic[model_key]["name"] #  "test-model" # 

# Set task
task = model_dic[model_key]["task"]

# Load a model
model = YOLO(model=model_path, task=task)#.load(model_path)  # build from YAML and transfer weights 
# model = YOLO("yolov8s.yaml").load("/home/ubuntu/repo/ultralytics/runs/detect/wiwynn-GPU_pins-20241205-/weights/best.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data=data, 
    cfg=cfg_path, 
    name=name, 
    # batch= 1000,
    # epochs=10
    )

