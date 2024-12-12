import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import os

# 确保设备正确
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 定义简单的 CNN 模型
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


# 2. 数据预处理和加载 CIFAR-10 数据集
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# 加载训练集、验证集和测试集
trainset = datasets.CIFAR10(
    root=os.path.dirname(os.path.abspath(__file__)) + "/data",
    train=True,
    download=True,
    transform=transform,
)
train_size = int(0.8 * len(trainset))  # 80% 用于训练
val_size = len(trainset) - train_size  # 剩余 20% 用于验证
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# 测试集数据
testset = datasets.CIFAR10(
    root=os.path.dirname(os.path.abspath(__file__)) + "/data",
    train=False,
    download=True,
    transform=transform,
)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 3. 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)  # 将模型移到指定设备（GPU 或 CPU）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 4. 训练模型并验证
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # 进入训练模式
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到指定设备
        optimizer.zero_grad()  # 清空梯度

        outputs = model(inputs)  # 计算输出
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    # 在验证集上验证
    model.eval()  # 进入评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # 获取预测的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(
        f"Validation Loss: {val_loss / len(valloader)}, Validation Accuracy: {val_accuracy}%"
    )

# 5. 在测试集上测试模型
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss / len(testloader)}, Test Accuracy: {test_accuracy}%")

# 6. 保存训练后的模型参数
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_model.pth")
torch.save(model.state_dict(), save_path)
print("model saved!")
