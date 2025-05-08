import torch
from torch import nn

class EnhancedEMA(nn.Module):
    def __init__(self, channels, factor=8, dilation_rate=2):
        super(EnhancedEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        # 原始 1x1 卷积，用于基本特征抽取
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

        # 改进：使用扩展卷积代替原始的 3x3 卷积
        self.conv_dilated = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                      padding=dilation_rate, dilation=dilation_rate)

        # 新增的全局上下文卷积分支
        self.global_context_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. 生成全局上下文特征
        global_context = self.global_context_conv(self.agp(x))  # b, c, 1, 1

        # 2. 分组特征计算
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 使用 1x1 卷积进行基础特征抽取
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 基于 1x1 和 Sigmoid 加权的特征
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # 改进：使用扩展卷积替代原始 3x3 卷积
        x2 = self.conv_dilated(group_x)

        # 生成加权注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # 结合全局上下文特征，进行加权融合
        final_output = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return final_output + global_context.expand_as(final_output)  # 加入全局上下文特征


# 测试代码
if __name__ == '__main__':
    block = EnhancedEMA(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print('input_size:', input.size())
    print('output_size:', output.size())
