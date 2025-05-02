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
        self.mid_channels = in_channels // 2

        # 每组的3x3深度卷积和1x1点卷积
        self.dwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=min(4, in_channels), bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pwconv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dwconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=min(4, in_channels), bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.pwconv2 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.mid_channels)
        self.relu4 = nn.ReLU(inplace=True)

        # 最终1x1卷积
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu5 = nn.ReLU(inplace=True)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming初始化 for 卷积层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm初始化：gamma=1, beta=0
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
        # 通道分割
        x1_a, x1_b = x1[:, :self.mid_channels, :, :], x1[:, self.mid_channels:, :, :]
        x2_a, x2_b = x2[:, :self.mid_channels, :, :], x2[:, self.mid_channels:, :, :]

        # 组1：Concat(x1_a, x2_a)
        group1 = torch.cat([x1_a, x2_a], dim=1)
        group1 = self.dwconv1(group1)
        group1 = self.bn1(group1)
        group1 = self.relu1(group1)
        group1 = self.pwconv1(group1)
        group1 = self.bn2(group1)
        group1 = self.relu2(group1)

        # 组2：Concat(x1_b, x2_b)
        group2 = torch.cat([x1_b, x2_b], dim=1)
        group2 = self.dwconv2(group2)
        group2 = self.bn3(group2)
        group2 = self.relu3(group2)
        group2 = self.pwconv2(group2)
        group2 = self.bn4(group2)
        group2 = self.relu4(group2)

        # Concat和通道重排
        out = torch.cat([group1, group2], dim=1)
        out = self.channel_shuffle(out)

        # 最终1x1卷积
        out = self.final_conv(out)
        out = self.bn5(out)
        out = self.relu5(out)

        return out
if __name__ == '__main__':
    # 测试
    x1 = torch.randn(1, 16, 480, 320)
    x2 = torch.randn(1, 16, 480, 320)
    model = LightFusionBlock(16)
    output = model(x1, x2)
    print(output.shape)  # 应该是 (1, 16, 64, 64)