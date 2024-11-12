from ultralytics import YOLO
import os

home_path = os.path.expanduser("~")

# Load a model
model = YOLO('/home/ubuntu/repo/ultralytics/runs/detect/wiwynn-RPN_pins-20241028-5/weights/best.pt') # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category
