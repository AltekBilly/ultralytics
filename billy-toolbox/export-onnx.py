from ultralytics import YOLO
import onnx

task = 'classify'
name = 'wiwynn-GPU_pins_Classification-20241118-2'

# Load a model
model = YOLO(f"./runs/{task}/{name}/weights/best.pt")  # load a custom trained model

# Export the model
dynamic=False
opset=17
model.export(format="onnx", dynamic=dynamic, opset=opset)

# 加載導出的 ONNX 模型
model = onnx.load(f"./runs/{task}/{name}/weights/best.onnx")

# 查看當前的 IR 版本
print(f"Current IR Version: {model.ir_version}")
print(f"Opset Version: {model.opset_import[0].version}")


# 修改 IR 版本為 8
model.ir_version = 8

# 保存修改後的模型
save_name = name if name[-1] == '-' else f'{name}-'
onnx.save(model, f"./runs/{task}{name}/weights/{save_name}best-Opest{model.opset_import[0].version}-IR{model.ir_version}.onnx")

print(f"Current IR Version: {model.ir_version}")
print(f"Opset Version: {model.opset_import[0].version}")