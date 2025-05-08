import torch
import torch.nn as nn
from timm.models.layers import SqueezeExcite

# 定义带Batch Normalization的卷积层
class Conv2d_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

# 全局自注意力模块
class GlobalAttention(nn.Module):
    def __init__(self, channels):
        super(GlobalAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)
        return out + x  # 残差连接

# 带有全局注意力的RepViT块
class GlobalAttentionRepViTBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2, hidden_dim=80, use_se=True, use_hs=True):
        super(GlobalAttentionRepViTBlock, self).__init__()
        # Token Mixer: 3x3 DW卷积 + SE层 + 全局注意力
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                GlobalAttention(inp),  # 添加全局注意力
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                GlobalAttention(inp),  # 添加全局注意力
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
        # Channel Mixer
        self.channel_mixer = nn.Sequential(
            Conv2d_BN(oup, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GELU() if use_hs else nn.ReLU(),
            Conv2d_BN(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bn_weight_init=0),
        )

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x

# 测试代码
if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 64, 64)
    model = GlobalAttentionRepViTBlock(inp=32, oup=32, kernel_size=3, stride=1)
    output_tensor = model(input_tensor)
    print('GlobalAttentionRepViTBlock - input size:', input_tensor.size())
    print('GlobalAttentionRepViTBlock - output size:', output_tensor.size())
