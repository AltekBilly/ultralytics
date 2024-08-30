import torch

# 載入 TorchScript 模型
script_model = torch.jit.load('runs/altek_landmark/Altek_Landmark-FacialLandmark-test-20240705-stride64-qat-2/weights/best.pt')

# 初始化參數計數器
total_params = 0

# 迭代模型的所有參數
for module in script_model.named_modules():
    for param in module[1].named_modules():
        for _param in param[1].named_modules():
            if _param.named_parameters():
                total_params += param[1].numel()


print(f'模型的總參數量為：{total_params}')