import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from tqdm import tqdm
import math
import numpy as np


# LightEdgeBlock 模块（与之前一致）
class LightEdgeBlock(nn.Module):
    def __init__(self, in_channels):
        super(LightEdgeBlock, self).__init__()

        # 每个分支独立的 1x1 卷积，降维到 in_channels // 4
        self.conv1x1_v = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )
        self.conv1x1_h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )
        self.conv1x1_c = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )
        self.conv1x1_d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )

        # 定义固定算子（3x3 卷积核，使用 Sobel 算子）
        self.op_v = torch.tensor([[[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]]], dtype=torch.float32)  # 垂直边缘
        self.op_h = torch.tensor([[[[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]]]], dtype=torch.float32)  # 水平边缘
        self.op_c = torch.tensor([[[[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]]]], dtype=torch.float32)  # 十字形
        self.op_d = torch.tensor([[[[-1, 0, 1],
                                    [0, 0, 0],
                                    [1, 0, -1]]]], dtype=torch.float32)  # 对角线

        # 每个分支的固定算子卷积（3x3）
        self.conv_v = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1,
                                groups=in_channels // 4, bias=False)
        self.conv_h = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1,
                                groups=in_channels // 4, bias=False)
        self.conv_c = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1,
                                groups=in_channels // 4, bias=False)
        self.conv_d = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1,
                                groups=in_channels // 4, bias=False)

        # 设置固定算子权重
        self._set_fixed_weights()

        # 每个分支的批量归一化和激活
        self.bn_relu_v = nn.Sequential(nn.BatchNorm2d(in_channels // 4), nn.ReLU(True))
        self.bn_relu_h = nn.Sequential(nn.BatchNorm2d(in_channels // 4), nn.ReLU(True))
        self.bn_relu_c = nn.Sequential(nn.BatchNorm2d(in_channels // 4), nn.ReLU(True))
        self.bn_relu_d = nn.Sequential(nn.BatchNorm2d(in_channels // 4), nn.ReLU(True))

        # 3x3 卷积融合特征
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        # 1x1 卷积调整通道
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def _set_fixed_weights(self):
        with torch.no_grad():
            self.conv_v.weight.data = self.op_v.repeat(self.conv_v.weight.size(0), 1, 1, 1)
            self.conv_h.weight.data = self.op_h.repeat(self.conv_h.weight.size(0), 1, 1, 1)
            self.conv_c.weight.data = self.op_c.repeat(self.conv_c.weight.size(0), 1, 1, 1)
            self.conv_d.weight.data = self.op_d.repeat(self.conv_d.weight.size(0), 1, 1, 1)
            self.conv_v.weight.requires_grad = False
            self.conv_h.weight.requires_grad = False
            self.conv_c.weight.requires_grad = False
            self.conv_d.weight.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(self, x):
        residual = x
        x_v = self.conv1x1_v(x)
        x_h = self.conv1x1_h(x)
        x_c = self.conv1x1_c(x)
        x_d = self.conv1x1_d(x)
        x_v = self.conv_v(x_v)
        x_h = self.conv_h(x_h)
        x_c = self.conv_c(x_c)
        x_d = self.conv_d(x_d)
        x_v = self.bn_relu_v(x_v)
        x_h = self.bn_relu_h(x_h)
        x_c = self.bn_relu_c(x_c)
        x_d = self.bn_relu_d(x_d)
        x = torch.cat([x_v, x_h, x_c, x_d], dim=1)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = x + residual
        return x


# 边缘检测网络
class EdgeDetectionNet(nn.Module):
    def __init__(self, in_channels=3, num_blocks=2, hidden_channels=32):
        super(EdgeDetectionNet, self).__init__()
        # 输入卷积
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # LightEdgeBlock 堆叠
        self.blocks = nn.ModuleList([LightEdgeBlock(hidden_channels) for _ in range(num_blocks)])
        # 输出卷积，生成单通道边缘图
        self.output_conv = nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_conv(x)
        x = self.sigmoid(x)
        return x


# 数据集类
class BSDS500Dataset(Dataset):
    def __init__(self, lst_file, root_dir, sub_sample=2000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 读取 lst 文件
        with open(lst_file, 'r') as f:
            lines = f.readlines()
        self.pairs = [line.strip().split() for line in lines]
        # 采样 sub_sample 个样本
        if sub_sample > 0 and sub_sample < len(self.pairs):
            self.pairs = random.sample(self.pairs, sub_sample)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if len(self.pairs[idx]) == 2:  # 训练集
            img_path, gt_path = self.pairs[idx]
            img_path = os.path.join(self.root_dir, img_path)
            gt_path = os.path.join(self.root_dir, gt_path)
            image = Image.open(img_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')  # 单通道
        else:  # 测试集
            img_path = self.pairs[idx][0]
            img_path = os.path.join(self.root_dir, img_path)
            image = Image.open(img_path).convert('RGB')
            gt = None

        if self.transform:
            image = self.transform(image)
            if gt is not None:
                gt = transforms.ToTensor()(gt)
                gt = (gt > 0.5).float()  # 二值化
        return image, gt, img_path


# 主函数
def main():
    # 参数设置
    data_root = r"D:\rawcode\data\BSDS500_flipped_rotated_pad"
    train_lst = os.path.join(data_root, "image-train.lst")
    test_lst = os.path.join(data_root, "image-test.lst")
    output_dir = os.path.join(data_root, "predictions")
    sub_sample = 2000
    num_epochs = 10
    batch_size = 8
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    train_dataset = BSDS500Dataset(train_lst, data_root, sub_sample=sub_sample, transform=transform)
    test_dataset = BSDS500Dataset(test_lst, data_root, sub_sample=-1, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 初始化模型
    model = EdgeDetectionNet(in_channels=3, num_blocks=3, hidden_channels=64).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和测试
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for images, gts, _ in train_bar:
            images, gts = images.to(device), gts.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, gts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

        # 测试
        model.eval()
        os.makedirs(os.path.join(output_dir, f"epoch_{epoch + 1}"), exist_ok=True)
        test_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Test]")
        with torch.no_grad():
            for images, _, img_paths in test_bar:
                images = images.to(device)
                outputs = model(images)
                # 保存预测结果
                for output, img_path in zip(outputs, img_paths):
                    output = output.squeeze().cpu().numpy()  # [H, W]
                    output = (output * 255).astype(np.uint8)  # 转换为 0-255
                    img_name = os.path.basename(img_path)
                    save_path = os.path.join(output_dir, f"epoch_{epoch + 1}", img_name.replace('.jpg', '.png'))
                    Image.fromarray(output).save(save_path)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, "edge_detection_model.pth"))


if __name__ == "__main__":
    main()