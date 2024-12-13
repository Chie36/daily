import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import os


# 1. 定义与训练时相同的 CNN 模型结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10 共有 10 类

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 扁平化
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. 加载训练好的模型
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)  # 将模型移到指定设备
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()  # 切换到评估模式


# 3. 预处理输入图像
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.jpg")
image = Image.open(img_path)

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0).to(device)  # 增加 batch 维度并移到设备

# 4. 使用模型进行推理
with torch.no_grad():
    output = model(input_batch)

_, predicted = torch.max(output, 1)  # 获取最大概率的类别

# CIFAR-10 类别标签
class_names = [
    "PLANE",
    "CAR",
    "BIRD",
    "CAT",
    "DEER",
    "DOG",
    "FROG",
    "HORSE",
    "SHIP",
    "TRUCK",
]

# 5. 输出预测结果
print(f"It is a {class_names[predicted.item()]}!")
