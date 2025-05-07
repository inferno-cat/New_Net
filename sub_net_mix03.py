import torch
import torch.nn as nn
import torch.nn.functional as F
from Sub_Tools.AT_CMUNeXTBlock import MultiScaleConvBlock
from Sub_Tools.AT_UpSample import DySample_UP_Outchannels as AT_UpSample

class SkipFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SkipFusionBlock, self).__init__()
        # 分组卷积
        self.group_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=2)
        # 反向瓶颈设计的逐点卷积
        self.pointwise_conv1 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels * 2)
        self.gelu1 = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, encoder_features, decoder_features):
        # 将编码器特征和解码器特征直接拼接
        concat_features = torch.cat((encoder_features, decoder_features), dim=1)
        # 分组卷积提取特征
        group_features = self.group_conv1(concat_features)
        # 反向瓶颈设计的逐点卷积融合特征
        fusion_features = self.gelu1(self.bn1(self.pointwise_conv1(group_features)))
        fusion_features = self.gelu2(self.bn2(self.pointwise_conv2(fusion_features)))
        return fusion_features

class OrdinaryConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OrdinaryConvBlock, self).__init__()
        # 逐点卷积
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 批归一化
        self.bn = nn.BatchNorm2d(out_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.down(x)
        x = self.conv1x1(x)

        return x

class MixNet(nn.Module):
    def __init__(self, base_dim=16):
        super(MixNet, self).__init__()
        # self.block = CPDCBlock
        # self.block = MixBlock
        self.in_channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 4]
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, self.in_channels[0], 3, 1, 1, bias=False),
            nn.Conv2d(self.in_channels[0], self.in_channels[0], 3, 1, 1, bias=False),
        )

        self.CMU1 = MultiScaleConvBlock(self.in_channels[0], self.in_channels[0])
        self.CMU2 = MultiScaleConvBlock(self.in_channels[1], self.in_channels[1])
        self.CMU3 = MultiScaleConvBlock(self.in_channels[2], self.in_channels[2])
        self.CMU4 = MultiScaleConvBlock(self.in_channels[3], self.in_channels[3])

        self.ord1 = OrdinaryConvBlock(self.in_channels[0], self.in_channels[0])
        self.ord2 = OrdinaryConvBlock(self.in_channels[1], self.in_channels[1])
        self.ord3 = OrdinaryConvBlock(self.in_channels[2], self.in_channels[2])

        self.down1 = DownSample(self.in_channels[0], self.in_channels[1])
        self.down2 = DownSample(self.in_channels[1], self.in_channels[2])
        self.down3 = DownSample(self.in_channels[2], self.in_channels[3])

        self.ord4 = OrdinaryConvBlock(self.in_channels[3], self.in_channels[3])
        self.up1 = AT_UpSample(self.in_channels[3], self.in_channels[2])
        self.up2 = AT_UpSample(self.in_channels[2], self.in_channels[1])
        self.up3 = AT_UpSample(self.in_channels[1], self.in_channels[0])

        self.skip3 = SkipFusionBlock(self.in_channels[2] * 2, self.in_channels[2])
        self.skip2 = SkipFusionBlock(self.in_channels[1] * 2, self.in_channels[1])
        self.skip1 = SkipFusionBlock(self.in_channels[0] * 2, self.in_channels[0])

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.in_channels[0], self.in_channels[0], 3, 1, 1),
            nn.BatchNorm2d(self.in_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels[0], 1, 3, 1, 1),
            nn.ReLU(inplace=True),
        )


        #
        # self.stage1 = self._make_layer(self.block, self.in_channels[0], 4)
        # self.stage2 = self._make_layer(self.block, self.in_channels[1], 4)
        # self.stage3 = self._make_layer(self.block, self.in_channels[2], 4)
        # self.stage4 = self._make_layer(self.block, self.in_channels[3], 4)
        #
        # self.down2 = DownSample(self.in_channels[0], self.in_channels[1])
        # self.down3 = DownSample(self.in_channels[1], self.in_channels[2])
        # self.down4 = DownSample(self.in_channels[2], self.in_channels[3])
        #
        # self.mscm4 = MultiScaleContextModule(self.in_channels[3])
        # self.mscm3 = MultiScaleContextModule(self.in_channels[2])
        # self.mscm2 = MultiScaleContextModule(self.in_channels[1])
        # self.mscm1 = MultiScaleContextModule(self.in_channels[0])
        #
        # self.de3 = nn.Sequential(
        #     Decoder(self.in_channels[2]),
        #     MultiScaleFCAttention(self.in_channels[2]),
        # )
        # self.de2 = nn.Sequential(
        #     Decoder(self.in_channels[1]),
        #     MultiScaleFCAttention(self.in_channels[1]),
        # )
        # self.de1 = nn.Sequential(
        #     Decoder(self.in_channels[0]),
        #     MultiScaleFCAttention(self.in_channels[0]),
        # )
        #
        # # self.up4 = UpBlock(self.in_channels[3], self.in_channels[2])
        # # self.up3 = UpBlock(self.in_channels[2], self.in_channels[1])
        # # self.up2 = UpBlock(self.in_channels[1], self.in_channels[0])
        # self.up4 = AT_UpSample(self.in_channels[3], self.in_channels[2])
        # self.up3 = AT_UpSample(self.in_channels[2], self.in_channels[1])
        # self.up2 = AT_UpSample(self.in_channels[1], self.in_channels[0])
        #
        # self.convnext = nn.Sequential(
        #     BaseConv(3, self.in_channels[0], 3, 1, activation=nn.ReLU(inplace=True), use_bn=True),
        #     ConvNeXtV2_Block(self.in_channels[0]),
        # )
        #
        # self.output_layer = nn.Sequential(
        #     BaseConv(2 * self.in_channels[0], self.in_channels[0], 1, 1, activation=nn.ReLU(inplace=True)),
        #     BaseConv(self.in_channels[0], 1, 3, 1),
        # )

    def _make_layer(self, block, dim, block_nums):
        layers = []

        for i in range(0, block_nums):
            layers.append(block(dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem_conv(x)
        x1 = self.ord1(self.CMU1(x))
        x2 = self.down1(x1)
        x2 = self.ord2(self.CMU2(x2))
        x3 = self.down2(x2)
        x3 = self.ord3(self.CMU3(x3))
        x4 = self.down3(x3)
        x4 = self.ord4(self.CMU4(x4))
        d4 = self.up1(x4)
        d4 = self.skip3(x3, d4)
        d3 = self.up2(d4)
        d3 = self.skip2(x2, d3)
        d2 = self.up3(d3)
        d2 = self.skip1(x1, d2)
        d1 = self.ord1(self.CMU1(d2))
        d1 = self.ord1(d1)

        o = self.output_layer(d1)
        return o

    # def convert_to_standard_conv(self):
    #     for module in self.modules():
    #         if isinstance(module, CPDCBlock):
    #             module.convert_to_standard_conv()

    # def load_state_dict(self, state_dict, strict=False):
    #     model_state_dict = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in model_state_dict:
    #             model_state_dict[name].copy_(param)
    #         elif name.replace('standard_conv', 'op_type') in model_state_dict:
    #             # 处理已转换的卷积
    #             new_name = name.replace('standard_conv', 'op_type')
    #             model_state_dict[new_name].copy_(param)
    #     super().load_state_dict(model_state_dict, strict=False)
    #     self.convert_to_standard_conv()

if __name__ == "__main__":
    # net = ET_UHNet(16)
    net = MixNet()
    x = torch.randn(4, 3, 480, 320)
    y = net(x)
    print(y.shape)
