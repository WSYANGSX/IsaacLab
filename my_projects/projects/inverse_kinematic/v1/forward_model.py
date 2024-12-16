import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from typing import Any, Callable

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# 1.定义数据集类
class CustomDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        # 初始化代码，比如加载数据列表、文件路径等
        # 这里只是示例，实际使用时需要替换
        self.file_path = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.raw_data: Any = []

        self.load_data()

    def __len__(self):
        # 返回数据集中的样本数
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引idx返回样本数据和标签
        # 这里只是示例，实际使用时需要替换
        joint_pos, ee_pose = self.data[idx], self.targets[idx]

        if self.transform is not None:
            joint_pos = self.transform(joint_pos)

        if self.target_transform is not None:
            ee_pose = self.target_transform(ee_pose)

        return joint_pos, ee_pose

    def load_data(self):
        raw_data = []
        with open(self.file_path, "r") as f:
            for line in f:
                # 假设每行都是有效的，并且用空格分隔
                row = list(map(float, line.strip().split()))
                raw_data.append(row)
            self.raw_data = torch.tensor(raw_data, dtype=torch.float32)

        if self.train:
            n_samples = int(0.7 * len(self.raw_data))  # 计算70%的样本数量
            indices = np.random.choice(
                len(self.raw_data), n_samples, replace=False
            )  # 随机选择索引
            self.raw_data = self.raw_data[indices]  # 根据索引选择样本
        else:
            n_samples = int(0.3 * len(self.raw_data))  # 计算30%的样本数量
            indices = np.random.choice(
                len(self.raw_data), n_samples, replace=False
            )  # 随机选择索引
            self.raw_data = self.raw_data[indices]  # 根据索引选择样本

        self.data = self.raw_data[:, 0:7]
        self.targets = self.raw_data[:, 7:10] * 1000
        # self.targets[:, 9:12] = self.raw_data[:, 9:12] * 1000
        # euler_angles = euler_xyz_from_quat(targets[:, 3:7])  # 转换成欧拉角
        # a = euler_angles[0].view(-1, 1)
        # b = euler_angles[1].view(-1, 1)
        # c = euler_angles[2].view(-1, 1)
        # self.targets = torch.cat((pos_targets, a, b, c), dim=-1)


# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义模型层
        self.fc1 = nn.Linear(7, 512)
        self.bn1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.LayerNorm(1024)

        self.fc3 = nn.Linear(1024, 256)
        self.bn3 = nn.LayerNorm(256)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.LayerNorm(128)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.LayerNorm(64)

        self.fc6 = nn.Linear(64, 3)

    def forward(self, x):
        # 定义前向传播
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x


# 训练模型
def train_model(model, criterion, optimizer, num_epochs=100000):
    for epoch in range(num_epochs):
        # 每个epoch的训练和验证步骤
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for _, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()  # 梯度归零

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        if epoch % 100 == 0:
            torch.save(
                model,
                "/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/inverse_kinematic/v1/forward_model.pth",
            )


if __name__ == "__main__":
    # 加载数据集
    print("************ Load Dataset ****************")
    print("It may take few minutes.Please wait.....")
    start_time = time.time()

    train_dataset = CustomDataset(
        root="/home/yangxf/Ominverse_RL_platform/IsaacLab/my_project/rl_projects/inverse_kinematic/v1/raw_data.txt",
        train=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    end_time = time.time()

    print(f"Load complete. Time spend: {end_time - start_time}s")

    # 定义模型
    model = MyModel().to(device=device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    # # 初始化学习率调整器，假设我们每1000个epoch将学习率乘以0.8
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    # 调用训练函数
    train_model(model, criterion, optimizer)
