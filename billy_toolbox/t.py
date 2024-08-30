import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, num_samples, num_channels, height, width, num_classes):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机图像和标签
        image = torch.randn(self.num_channels, self.height, self.width)
        label = torch.randint(0, self.num_classes, (self.height, self.width))  # 修改为二维标签
        return image, label

# 创建数据集和数据加载器
num_samples = 1000
num_channels = 1
height = 28
width = 28
num_classes = 10
batch_size = 32

dataset = RandomDataset(num_samples, num_channels, height, width, num_classes)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=2, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = DepthwiseSeparableConv(1, 32)
        self.conv2 = DepthwiseSeparableConv(32, 16)
        layers =[self.conv1, self.conv2]
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = self.m(x)
        return x

# 创建模型实例
conv1 = DepthwiseSeparableConv(1, 32)
conv2 = DepthwiseSeparableConv(32, 16)
nn.Sequential()
# model = Model()

# 为模型设置量化配置
# model.qconfig =  quant.get_default_qat_qconfig('qnnpack')
conv1.qconfig =  quant.get_default_qat_qconfig('qnnpack')
conv2.qconfig =  quant.get_default_qat_qconfig('qnnpack')

# 使用torch.quantization对模型进行量化感知训练准备
# model_prepared = quant.prepare_qat(model)
conv1_prepared = quant.prepare_qat(conv1)
conv2_prepared = quant.prepare_qat(conv2)
layers = [conv1_prepared, conv2_prepared]
model_prepared = nn.Sequential(*layers)

# 假设有训练数据
# optimizer = torch.optim.SGD(model_prepared.parameters(), lr=0.01)
# loss_fn = nn.CrossEntropyLoss()

# # 训练循环
# num_epochs = 10
# for epoch in range(num_epochs):
#     for input, target in data_loader:
#         optimizer.zero_grad()
#         output = model_prepared(input)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()

# print(model)
# 转换为量化模型
model_int8 = quant.convert(model_prepared)

# 导出量化模型
torch.save(model_int8.state_dict(), 'quantized_model.pth')