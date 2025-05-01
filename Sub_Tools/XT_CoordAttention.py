import torch
import torch.nn as nn
import math

class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        """
        CoordAttention 模块
        Args:
            in_channels (int): 输入通道数
            reduction (int): 通道降维比例，默认为 32
        """
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向池化

        # 中间通道数
        mid_channels = max(8, in_channels // reduction)

        # 共享的 1x1 卷积层，用于处理拼接后的特征
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        # 生成水平和垂直方向的注意力权重，输出通道数与输入一致
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入特征图，形状 (batch_size, in_channels, H, W)
        Returns:
            torch.Tensor: 加权后的特征图，形状 (batch_size, in_channels, H, W)
        """
        identity = x  # 保存输入用于残差连接
        n, c, h, w = x.size()

        # 水平方向池化: (batch_size, c, h, 1)
        x_h = self.pool_h(x)
        # 垂直方向池化: (batch_size, c, 1, w)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 转置为 (batch_size, c, w, 1)

        # 拼接水平和垂直特征
        y = torch.cat([x_h, x_w], dim=2)  # (batch_size, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 拆分为水平和垂直方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 转置回 (batch_size, mid_channels, 1, w)

        # 生成注意力权重
        a_h = self.conv_h(x_h)  # (batch_size, in_channels, h, 1)
        a_w = self.conv_w(x_w)  # (batch_size, in_channels, 1, w)
        a_h = self.sigmoid(a_h)
        a_w = self.sigmoid(a_w)

        # 加权输入特征图
        out = identity * a_h * a_w

        return out

# 示例使用
if __name__ == "__main__":
    # 随机输入特征图: (batch_size=2, channels=16, height=240, width=160)
    x = torch.randn(2, 16, 240, 160)
    # 初始化 CoordAttention 模块
    ca = CoordAttention(in_channels=16, reduction=32)
    # 前向传播
    out = ca(x)
    print(out.shape)  # 输出形状: torch.Size([2, 16, 240, 160])