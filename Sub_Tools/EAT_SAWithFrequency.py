import torch
import torch.nn as nn
import torch.fft as fft

class SpatialAttentionWithFrequency(nn.Module):
    def __init__(self, in_channels):
        """
        使用频域处理增强空间注意力图
        :param in_channels: 输入的通道数
        """
        super(SpatialAttentionWithFrequency, self).__init__()

        # 空间注意力卷积层
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3)

        # Sigmoid激活生成空间注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播，生成通过频域处理增强的空间注意力图
        :param x: 输入特征图
        :return: 生成的空间注意力图
        """
        batch_size, channels, height, width = x.size()

        # 时域空间注意力图生成（通过卷积）
        spatial_attention = self.conv1(x)

        # 使用傅里叶变换提取频域信息
        x_freq = fft.fft2(x)  # 进行二维傅里叶变换
        x_freq = torch.abs(x_freq)  # 获取频域的幅度谱（幅度可以反映细节信息）

        # 将频域信息与时域注意力图相结合
        spatial_attention = spatial_attention + x_freq.mean(dim=1, keepdim=True)  # 平均频域信息，保持通道维度

        # 使用Sigmoid激活生成空间注意力图
        attention_map = self.sigmoid(spatial_attention)

        # 按照注意力图加权输入
        output = x * attention_map
        return output


# 测试代码
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 32, 32)  # batch_size=8, channels=64, height=32, width=32
    spatial_attention_freq = SpatialAttentionWithFrequency(in_channels=64)
    output_tensor = spatial_attention_freq(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
