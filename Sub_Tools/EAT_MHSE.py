import torch
import torch.nn as nn

class MultiHeadSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, num_heads=4):
        """
        多头通道注意力机制模块
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        :param num_heads: 注意力头的数量
        """
        super(MultiHeadSEBlock, self).__init__()
        assert in_channels % num_heads == 0, "通道数必须能被注意力头数量整除"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # 每个头的通道数
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

        # 确保降维后的通道数至少为 1
        reduced_dim = max(1, self.head_dim // reduction)

        self.conv1 = nn.Conv2d(self.head_dim, reduced_dim, kernel_size=1, bias=False)  # 降维
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduced_dim, self.head_dim, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入维度
        # 分割通道为多个头
        x_split = x.view(b * self.num_heads, self.head_dim, h, w)  # 将通道分为多个头，每个头作为一个批次
        # 对每个头进行操作
        y = self.global_avg_pool(x_split)  # 全局平均池化，形状 [b*num_heads, head_dim, 1, 1]
        y = self.conv1(y)  # 降维
        y = self.relu(y)
        y = self.conv2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成注意力权重
        # 将注意力权重应用到输入张量
        x_split = x_split * y  # 对每个头的特征加权
        # 恢复形状
        x_out = x_split.view(b, c, h, w)  # 合并所有头
        return x_out

# 测试 MultiHeadSEBlock
if __name__ == "__main__":
    # 输入张量，形状为 [batch_size, channels, height, width]
    input_tensor = torch.randn(8, 64, 32, 32)  # 批量大小8，通道数64，特征图尺寸32x32
    se_block = MultiHeadSEBlock(in_channels=64, reduction=16, num_heads=4)  # 实例化多头SE模块
    output_tensor = se_block(input_tensor)  # 前向传播
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
