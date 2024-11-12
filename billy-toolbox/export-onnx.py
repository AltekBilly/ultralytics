from ultralytics import YOLO

# Load a model
model = YOLO("./runs/detect/wiwynn-RPN_pins-20241028-5/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")