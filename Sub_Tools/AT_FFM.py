import torch
import torch.nn as nn


class InnovativeDFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 多尺度平均池化
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(4)

        # 注意力卷积层，使用 2D 卷积
        self.conv_atten1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_atten2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_atten3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 门控机制
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 通道减少卷积层，使用 2D 卷积
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # 残差分支
        self.residual_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # 两个 2D 卷积层用于计算注意力
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        # Sigmoid 激活函数
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        # 沿着通道维度拼接输入特征
        output = torch.cat([x, skip], dim=1)

        # 多尺度平均池化及注意力计算
        att1 = self.conv_atten1(self.avg_pool1(output))
        att2 = self.conv_atten2(self.avg_pool2(output))
        att3 = self.conv_atten3(self.avg_pool3(output))

        # 上采样使尺寸一致
        att2 = nn.functional.interpolate(att2, size=att1.size()[2:], mode='bilinear', align_corners=True)
        att3 = nn.functional.interpolate(att3, size=att1.size()[2:], mode='bilinear', align_corners=True)

        # 融合多尺度注意力
        att = att1 + att2 + att3

        # 门控机制
        gate = self.gate_conv(output)
        att = att * gate

        # 应用注意力权重
        output = output * att

        # 减少通道数量
        output = self.conv_redu(output)

        # 残差分支
        residual = self.residual_conv(x)
        output = output + residual

        # 计算另一个注意力权重
        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)

        # 应用另一个注意力权重
        output = output * att
        return output

if __name__ == '__main__':
    # 生成随机输入数据，2D 图像维度 (B, C, H, W)
    input1 = torch.randn(1, 32,480, 320)
    input2 = torch.randn(1, 32,480, 320)
    # 初始化创新 DFF 模块
    model = InnovativeDFF(32)
    # 前向传播
    output = model(input1, input2)
    print("InnovativeDFF_input size:", input1.size())
    print("InnovativeDFF_Output size:", output.size())