import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCAttention(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(DFCAttention, self).__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向的平均池化
        self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))  # 水平方向的最大池化
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向的平均池化
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))  # 垂直方向的最大池化

        mip = max(8, inp // reduction)  # 计算中间通道数

        self.conv1_h = nn.Conv2d(inp * 2, mip, kernel_size=1, stride=1, padding=0)  # 水平方向的卷积
        self.bn1_h = nn.BatchNorm2d(mip)
        self.act_h = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 水平方向的卷积

        self.conv1_w = nn.Conv2d(inp * 2, mip, kernel_size=1, stride=1, padding=0)  # 垂直方向的卷积
        self.bn1_w = nn.BatchNorm2d(mip)
        self.act_w = nn.Hardswish()
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 垂直方向的卷积

    def forward(self, x):
        identity = x  # 保存输入特征，用于后续融合

        # 水平方向的特征聚合
        avg_x_h = self.avg_pool_h(x)
        max_x_h = self.max_pool_h(x)
        x_h = torch.cat([avg_x_h, max_x_h], dim=1)  # 沿通道维度拼接两种池化结果
        x_h = self.conv1_h(x_h)
        x_h = self.bn1_h(x_h)
        x_h = self.act_h(x_h)
        x_h = self.conv_h(x_h)
        x_h = F.hardsigmoid(x_h)

        # 垂直方向的特征聚合
        avg_x_w = self.avg_pool_w(x)
        max_x_w = self.max_pool_w(x)
        x_w = torch.cat([avg_x_w, max_x_w], dim=1)  # 沿通道维度拼接两种池化结果
        x_w = self.conv1_w(x_w)
        x_w = self.bn1_w(x_w)
        x_w = self.act_w(x_w)
        x_w = self.conv_w(x_w)
        x_w = F.hardsigmoid(x_w)

        # 注意力图的融合
        attention = x_h * x_w

        # 加权融合特征图
        out = identity * attention

        return out

# 测试代码
if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 480, 320)
    dcf_attention = DFCAttention(32, 32)
    output_tensor = dcf_attention(input_tensor)
    print(output_tensor.shape)  # 输出特征图的形状