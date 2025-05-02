import torch
import torch.nn as nn
import torch.nn.init as init


class FastConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=-1, use_final_conv=True):
        super(FastConvBlock, self).__init__()
        # 如果out_channels为-1，设置为in_channels
        if out_channels == -1:
            out_channels = in_channels
        assert in_channels == out_channels, "Input and output channels must be equal"
        self.mid_channels = in_channels // 2

        # 卷积支：3x3深度卷积 + 1x1点卷积
        self.dwconv = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1,
                                groups=self.mid_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pwconv = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 可选1x1卷积
        self.use_final_conv = use_final_conv
        if use_final_conv:
            self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.relu3 = nn.ReLU(inplace=True)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming初始化 for 卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # bias=False，无需初始化bias
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm初始化：gamma=1, beta=0
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)
                # 运行均值和方差初始化（PyTorch默认已处理，但明确设置）
                init.constant_(m.running_mean, 0.0)
                init.constant_(m.running_var, 1.0)

    def channel_shuffle(self, x, groups=2):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        # 通道分割
        identity, conv = x[:, :self.mid_channels, :, :], x[:, self.mid_channels:, :, :]

        # 卷积支
        conv = self.dwconv(conv)
        conv = self.bn1(conv)
        conv = self.relu1(conv)
        conv = self.pwconv(conv)
        conv = self.bn2(conv)
        conv = self.relu2(conv)

        # Concat
        out = torch.cat([identity, conv], dim=1)

        # 通道重排
        out = self.channel_shuffle(out)

        # 可选1x1卷积
        if self.use_final_conv:
            out = self.final_conv(out)
            out = self.bn3(out)
            out = self.relu3(out)

        return out
class FastConvList(nn.Module):
    def __init__(self, in_channels, size, use_final_conv=True):
        super(FastConvList, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(size):
            self.blocks.append(FastConvBlock(in_channels, in_channels, use_final_conv))
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':
    # 输入输出通道数均为64
    model = FastConvBlock(in_channels=32, use_final_conv=True)
    x = torch.randn(1, 32, 480, 320)
    y = model(x)
    print(y.shape)  # torch.Size([1, 64, 32, 32])

    model2 = FastConvList(in_channels=32, size=4, use_final_conv=True)
    x2 = torch.randn(1, 32, 480, 320)
    y2 = model2(x2)
    print(y2.shape)  # torch.Size([1, 64, 32, 32])