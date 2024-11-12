from ultralytics import YOLO
import os

home_path = os.path.expanduser("~")

date = "20241112"


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
}

model_key = "wiwynn-High_Value_Room"

# Set path
model_path = model_dic[model_key]["model"]
cfg_path = model_dic[model_key]["cfg"]
data = model_dic[model_key]["data"]
name = model_dic[model_key]["name"] #  "test-model" # 

# Set task
task = model_dic[model_key]["task"]

# Load a model
model = YOLO(model=model_path, task=task)#.load(model_path)  # build from YAML and transfer weights 
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data=data, 
    cfg=cfg_path, 
    name=name, 
    # batch= 1000,
    # epochs=10
    )

