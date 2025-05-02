import torch
import torch.nn as nn
import torch.nn.init as init


class LightFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=-1):
        super(LightFusionBlock, self).__init__()
        # 如果out_channels为-1，设置为in_channels
        if out_channels == -1:
            out_channels = in_channels
        assert out_channels == in_channels, "Output channels must equal input channels"
        self.in_channels = in_channels

        # 3x3深度卷积（输入2C）
        self.dwconv = nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, padding=1, groups=2 * in_channels,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(2 * in_channels)
        # 移除深度卷积后的ReLU，减少元素级操作
        self.pwconv = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)
                init.constant_(m.running_mean, 0.0)
                init.constant_(m.running_var, 1.0)

    def channel_shuffle(self, x, groups=2):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x1, x2):
        # 初始Concat
        out = torch.cat([x1, x2], dim=1)  # 形状：(2C, H, W)

        # 通道重排
        out = self.channel_shuffle(out)

        # 3x3深度卷积
        out = self.dwconv(out)
        out = self.bn1(out)
        # 无ReLU，减少元素级操作

        # 1x1点卷积
        out = self.pwconv(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
if __name__ == "__main__":
    # 测试代码
    x1 = torch.randn(1, 16, 32, 32)
    x2 = torch.randn(1, 16, 32, 32)
    block = LightFusionBlock(16)
    output = block(x1, x2)
    print(output.shape)  # 应该是 (1, 16, 32, 32)