import torch
import torch.nn as nn
from sub_edge001 import CPDCBlock
from Sub_Tools.ET_PDDPBlock import PDDPBlock as ET_PDDPBlock
from Sub_Tools.AT_UpSample import DySample_UP_Outchannels as AT_UpSample

class BaseConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True, relu = True, bias=False):
        super(BaseConv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

# Depthwise Separable Convolution
class BaseConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, bn=True, relu=True):
        super(BaseConvDW, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class FBlock(nn.Module):
    def __init__(self, in_channels, M, out_channels, upsample=True):
        super(FBlock, self).__init__()
        if upsample:
            self.up = AT_UpSample(in_channels, in_channels)
        self.conv1x1_1 = BaseConv1x1(in_channels, M, bn=True, relu=True)
        self.Dconv1 = BaseConvDW(M, M, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
        self.Dconv2 = BaseConvDW(M, M, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
        self.bn2 = nn.BatchNorm2d(M)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1x1_2 = BaseConv1x1(M, out_channels, bn=False, relu=False)
    def forward(self, x):
        if hasattr(self, 'up'):
            x = self.up(x)
        x = self.conv1x1_1(x)
        x = self.Dconv1(x)
        x = self.Dconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv1x1_2(x)
        return x

class PoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(PoolBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.max_pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)

        if self.out_channels == self.in_channels:
            x_out = max_pooled + avg_pooled
        else:
            x_out = torch.cat((max_pooled, avg_pooled), dim=1)
            assert x_out.shape[1] == self.out_channels, f"Expected {self.out_channels} channels, got {x_out.shape[1]}"
        return x_out


class ET_UHNet(nn.Module):
    def __init__(self, M=16,):
        super(ET_UHNet, self).__init__()
        self.in_channels = 3
        self.out_channels = 1

        self.conv1x1 = BaseConv1x1(self.in_channels, M, bn=True, relu=True)
        self.PDDPList1 = nn.Sequential(
            # ET_PDDPBlock(M, M // 2, M),
            # ET_PDDPBlock(M, M // 2, M),
            # ET_PDDPBlock(M, M // 2, M),
            # ET_PDDPBlock(M, M // 2, M),
            ET_PDDPBlock(M, M * 2 , M),
            ET_PDDPBlock(M, M * 2 , M),
            ET_PDDPBlock(M, M * 2 , M),
            ET_PDDPBlock(M, M * 2 , M),

        )
        self.poolblock1 = PoolBlock(M, M * 2, kernel_size=2, stride=2)
        self.PDDPList2 = nn.Sequential(
            # ET_PDDPBlock(M * 2, M, M * 2),
            # ET_PDDPBlock(M * 2, M, M * 2),
            # ET_PDDPBlock(M * 2, M, M * 2),
            # ET_PDDPBlock(M * 2, M, M * 2),
            ET_PDDPBlock(M * 2, M * 4 , M * 2),
            ET_PDDPBlock(M * 2, M * 4 , M * 2),
            ET_PDDPBlock(M * 2, M * 4 , M * 2),
            ET_PDDPBlock(M * 2, M * 4 , M * 2),
        )
        self.poolblock2 = PoolBlock(M * 2, M * 4, kernel_size=2, stride=2)
        self.PDDPList3 = nn.Sequential(
            # ET_PDDPBlock(M * 4, M * 2, M * 4),
            # ET_PDDPBlock(M * 4, M * 2, M * 4),
            # ET_PDDPBlock(M * 4, M * 2, M * 4),
            # ET_PDDPBlock(M * 4, M * 2, M * 4),
            ET_PDDPBlock(M * 4, M * 8 , M * 4),
            ET_PDDPBlock(M * 4, M * 8 , M * 4),
            ET_PDDPBlock(M * 4, M * 8 , M * 4),
            ET_PDDPBlock(M * 4, M * 8 , M * 4),
        )

        # self.fblock1 = FBlock(M * 4, M * 2, M * 2)
        # self.fblock2 = FBlock(M * 2, M, M)
        # # self.fblock3 = FBlock(M, M // 2, 1, upsample=False)
        # self.fblock3 = FBlock(M, M // 2, M, upsample=False)
        #
        # self.e1 = CPDCBlock(M)
        # self.e2 = CPDCBlock(M * 2)
        # self.e3 = CPDCBlock(M * 4)
        #
        # self.d1 = CPDCBlock(M * 2)
        # self.d2 = CPDCBlock(M )
        #
        # self.fb_cpdc_3 = FBlock(M * 4, M * 2, M * 2)
        # self.fb_cpdc_2 = FBlock(M * 6, M, M)
        # self.fb_cpdc_1 = FBlock(M * 3, M // 2, M, upsample=False)
        #
        # # fuselayer将fblock3与fb_cpdc_3的输出进行融合，从两个M通道的特征图获取单通道输出的特征图，先用1*1将通道数减半变为M，然后用3*3卷积将通道数变为1
        # self.fuselayer = nn.Sequential(
        #     nn.Conv2d(M * 2, M, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Conv2d(M, 1, kernel_size=3, stride=1, padding=1, bias=False),
        # )

        self.fblock1 = FBlock(M * 4, M * 8, M * 2)
        self.fblock2 = FBlock(M * 2, M *4 , M)
        # self.fblock3 = FBlock(M, M // 2, 1, upsample=False)
        self.fblock3 = FBlock(M, M*2, M, upsample=False)

        self.e1 = CPDCBlock(M)
        self.e2 = CPDCBlock(M * 2)
        self.e3 = CPDCBlock(M * 4)

        self.d1 = CPDCBlock(M * 2)
        self.d2 = CPDCBlock(M )

        self.fb_cpdc_3 = FBlock(M * 4, M * 8, M * 2)
        self.fb_cpdc_2 = FBlock(M * 6, M * 12, M)
        self.fb_cpdc_1 = FBlock(M * 3, M *6 , M, upsample=False)

        # fuselayer将fblock3与fb_cpdc_3的输出进行融合，从两个M通道的特征图获取单通道输出的特征图，先用1*1将通道数减半变为M，然后用3*3卷积将通道数变为1
        self.fuselayer = nn.Sequential(
            nn.Conv2d(M * 2, M, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(M, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )


    def forward(self, x):
        x = self.conv1x1(x)
        pddp1 = self.PDDPList1(x)
        pddp2 = self.PDDPList2(self.poolblock1(pddp1))
        pddp3 = self.PDDPList3(self.poolblock2(pddp2))

        f1 = self.fblock1(pddp3)
        f1_c = f1 + pddp2
        f2 = self.fblock2(f1_c)
        f2_c = f2 + pddp1
        f3 = self.fblock3(f2_c)

        de1 = self.fb_cpdc_3(pddp3)
        de2 = self.fb_cpdc_2(torch.cat((pddp2, f1, de1), dim=1))
        de3 = self.fb_cpdc_1(torch.cat((pddp1, f2, de2), dim=1))

        o = self.fuselayer(torch.cat((f3, de3), dim=1))
        return torch.sigmoid(o)


        # return torch.sigmoid(f3)

    def convert_to_standard_conv(self):
        for module in self.modules():
            if isinstance(module, CPDCBlock):
                module.convert_to_standard_conv()

    def load_state_dict(self, state_dict, strict=False):
        model_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            elif name.replace('standard_conv', 'op_type') in model_state_dict:
                # 处理已转换的卷积
                new_name = name.replace('standard_conv', 'op_type')
                model_state_dict[new_name].copy_(param)
        super().load_state_dict(model_state_dict, strict=False)
        self.convert_to_standard_conv()

# class PDCNet(nn.Module):
#     def __init__(self, base_dim=16):
#         super(PDCNet, self).__init__()
#         self.block = CPDCBlock
#         # self.block = MixBlock
#         # self.in_channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 4]
#         # self.stem_conv = nn.Sequential(
#         #     nn.Conv2d(3, self.in_channels[0], 3, 1, 1, bias=False),
#         #     nn.Conv2d(self.in_channels[0], self.in_channels[0], 3, 1, 1, bias=False),
#         # )
#         #
#         # self.stage1 = self._make_layer(self.block, self.in_channels[0], 4)
#         # self.stage2 = self._make_layer(self.block, self.in_channels[1], 4)
#         # self.stage3 = self._make_layer(self.block, self.in_channels[2], 4)
#         # self.stage4 = self._make_layer(self.block, self.in_channels[3], 4)
#         #
#         # self.down2 = DownSample(self.in_channels[0], self.in_channels[1])
#         # self.down3 = DownSample(self.in_channels[1], self.in_channels[2])
#         # self.down4 = DownSample(self.in_channels[2], self.in_channels[3])
#         #
#         # self.mscm4 = MultiScaleContextModule(self.in_channels[3])
#         # self.mscm3 = MultiScaleContextModule(self.in_channels[2])
#         # self.mscm2 = MultiScaleContextModule(self.in_channels[1])
#         # self.mscm1 = MultiScaleContextModule(self.in_channels[0])
#         #
#         # self.de3 = nn.Sequential(
#         #     Decoder(self.in_channels[2]),
#         #     MultiScaleFCAttention(self.in_channels[2]),
#         # )
#         # self.de2 = nn.Sequential(
#         #     Decoder(self.in_channels[1]),
#         #     MultiScaleFCAttention(self.in_channels[1]),
#         # )
#         # self.de1 = nn.Sequential(
#         #     Decoder(self.in_channels[0]),
#         #     MultiScaleFCAttention(self.in_channels[0]),
#         # )
#         #
#         # # self.up4 = UpBlock(self.in_channels[3], self.in_channels[2])
#         # # self.up3 = UpBlock(self.in_channels[2], self.in_channels[1])
#         # # self.up2 = UpBlock(self.in_channels[1], self.in_channels[0])
#         # self.up4 = AT_UpSample(self.in_channels[3], self.in_channels[2])
#         # self.up3 = AT_UpSample(self.in_channels[2], self.in_channels[1])
#         # self.up2 = AT_UpSample(self.in_channels[1], self.in_channels[0])
#         #
#         # self.convnext = nn.Sequential(
#         #     BaseConv(3, self.in_channels[0], 3, 1, activation=nn.ReLU(inplace=True), use_bn=True),
#         #     ConvNeXtV2_Block(self.in_channels[0]),
#         # )
#         #
#         # self.output_layer = nn.Sequential(
#         #     BaseConv(2 * self.in_channels[0], self.in_channels[0], 1, 1, activation=nn.ReLU(inplace=True)),
#         #     BaseConv(self.in_channels[0], 1, 3, 1),
#         # )
#
#     def _make_layer(self, block, dim, block_nums):
#         layers = []
#
#         for i in range(0, block_nums):
#             layers.append(block(dim))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # convnext = self.convnext(x)
#         #
#         # conv_stem = self.stem_conv(x)
#         #
#         # conv1 = self.stage1(conv_stem)  # C
#         # conv2 = self.stage2(self.down2(conv1))  # 2C
#         # conv3 = self.stage3(self.down3(conv2))  # 4C
#         # conv4 = self.stage4(self.down4(conv3))  # 4C
#         #
#         # mscm4 = self.mscm4(conv4)
#         # mscm4_up = self.up4(mscm4)  # 4C
#         # mscm3 = self.mscm3(conv3)
#         # de3 = self.de3(mscm3 + mscm4_up)
#         # de3_up = self.up3(de3)  # 2C
#         # mscm2 = self.mscm2(conv2)
#         # de2 = self.de2(mscm2 + de3_up)
#         # de2_up = self.up2(de2)  # C
#         # mscm1 = self.mscm1(conv1)
#         # de1 = self.de1(mscm1 + de2_up)
#         #
#         # output = self.output_layer(torch.cat([de1, convnext], dim=1))
#         #
#         # return torch.sigmoid(output)
#         pass
#
#     def convert_to_standard_conv(self):
#         for module in self.modules():
#             if isinstance(module, CPDCBlock):
#                 module.convert_to_standard_conv()
#
#     def load_state_dict(self, state_dict, strict=False):
#         model_state_dict = self.state_dict()
#         for name, param in state_dict.items():
#             if name in model_state_dict:
#                 model_state_dict[name].copy_(param)
#             elif name.replace('standard_conv', 'op_type') in model_state_dict:
#                 # 处理已转换的卷积
#                 new_name = name.replace('standard_conv', 'op_type')
#                 model_state_dict[new_name].copy_(param)
#         super().load_state_dict(model_state_dict, strict=False)
#         self.convert_to_standard_conv()

if __name__ == "__main__":
    net = ET_UHNet(16)
    x = torch.randn(4, 3, 480, 320)
    y = net(x)
    print(y.shape)
