import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from edge import Conv2d, EdgeConv, CPDCBlock, PlainBlock
from timm.models.layers import trunc_normal_, DropPath
from Sub_Tools.AT_UpSample import DySample_UP_Outchannels as AT_UpSample
from Sub_Tools.AT_DownSample import WTFDown as AT_DownSample
from Sub_Tools.AT_MSB import MSPABlock as AT_MSB
from Sub_Tools.AT_WTConv import WTConv2d as AT_WTConv
from Sub_Tools.AT_CatFuseBlock import CloMSFM as AT_CatFuse
from Sub_Tools.AT_LCA import LCA as AT_LCA
from Sub_Tools.AT_CBAM import CBAM as AT_CBAM
from Sub_Tools.AT_CBAM import EAT_CBAM as EAT_CBAM
from Sub_Tools.AT_SESA import SEWithAdvancedSpatialAttention as AT_SESA
from Sub_Tools.AT_LWGA import DynamicLWGA_Block as AT_LWGA
from Sub_Tools.AT_SMFA import SMFADynamicDownscale as AT_SMFA
from Sub_Tools.AT_SSA import SAABlocklist as AT_SSA
# from Sub_Tools.ET_PDDP_PDBlock import PDDP_FourDiffBlock as PD_PDDPBlock
# from Sub_Tools.ET_PDDP_PDBlock import PDDP_DiffBlock as PD_PDDPBlock
from Sub_Tools.ET_NewNet import BaseConv1x1, BaseConvDW, FBlock, PoolBlock
from pdc_attention_network import BaseConv, ConvNeXtV2_Block, Decoder
from Sub_Tools.ET_PDDPBlock import PDDPBlock as ET_PDDPBlock

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    def forward(self, x):
        x = self.down(x)
        x = self.conv1x1(x)
        return x

class ET_MSEM(nn.Module):
    def __init__(self, dim):
        super(ET_MSEM, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 3, dilation=3),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 3, 1, 4, dilation=4),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = BaseConv1x1(dim, dim, bn=False, relu=False)
        self.attn = AT_SESA(dim, reduction=16)
    def forward(self, x):
        residual = x
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        br3 = self.branch3(x)
        br4 = self.branch4(x)
        x = torch.cat([br1, br2, br3, br4], dim=1)
        x = self.conv1x1(x)
        x = self.attn(x)
        x = x + residual
        return x

def make_PDDPList(in_channels, M=-1, num_blocks=4):
    if M == -1:
        M = in_channels // 2
    pddp_list = []
    for i in range(num_blocks):
        pddp_list.append(ET_PDDPBlock(in_channels, M, in_channels))
    return nn.Sequential(*pddp_list)

class MixNet(nn.Module):
    def __init__(self, base_dim=8, ):
        super(MixNet, self).__init__()
        self.block = CPDCBlock
        self.in_channels = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, self.in_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels[0]),
            nn.Conv2d(self.in_channels[0], self.in_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channels[0]),
        )

        self.stage1 = self._make_layer(self.block, self.in_channels[0], 4)
        self.stage2 = self._make_layer(self.block, self.in_channels[1], 4)
        self.stage3 = self._make_layer(self.block, self.in_channels[2], 4)
        self.stage4 = self._make_layer(self.block, self.in_channels[3], 4)

        self.down2 = DownSample(self.in_channels[0], self.in_channels[1])
        self.down3= DownSample(self.in_channels[1], self.in_channels[2])
        self.down4 = DownSample(self.in_channels[2], self.in_channels[3])

        self.mscm4 = ET_MSEM(self.in_channels[3])
        self.mscm3 = ET_MSEM(self.in_channels[2])
        self.mscm2 = ET_MSEM(self.in_channels[1])
        self.mscm1 = ET_MSEM(self.in_channels[0])

        self.up4 = AT_UpSample(self.in_channels[3], self.in_channels[2])
        self.up3 = AT_UpSample(self.in_channels[2], self.in_channels[1])
        self.up2 = AT_UpSample(self.in_channels[1], self.in_channels[0])

        self.de3 = Decoder(self.in_channels[2])
        self.de2 = Decoder(self.in_channels[1])
        self.de1 = Decoder(self.in_channels[0])

        self.M = [base_dim, base_dim, base_dim, base_dim]
        # self.convM = BaseConv1x1(self.in_channels[0], self.M[0], bn=False, relu=False)
        self.pddp1 = make_PDDPList(self.in_channels[0], M = self.M[0], num_blocks=4)
        self.pddp2 = make_PDDPList(self.in_channels[1], M = self.M[1], num_blocks=4)
        self.pddp3 = make_PDDPList(self.in_channels[2], M = self.M[2], num_blocks=4)
        self.pddp4 = make_PDDPList(self.in_channels[3], M = self.M[3], num_blocks=4)

        self.down_pool2 = PoolBlock(self.in_channels[0], self.in_channels[1])
        self.down_pool3 = PoolBlock(self.in_channels[1], self.in_channels[2])
        self.down_pool4 = PoolBlock(self.in_channels[2], self.in_channels[3])

        self.down_lwga2 = AT_LWGA(self.in_channels[1], self.in_channels[1])
        self.down_lwga3 = AT_LWGA(self.in_channels[2], self.in_channels[2])
        self.down_lwga4 = AT_LWGA(self.in_channels[3], self.in_channels[3])

        self.up_fblock4 = FBlock(self.in_channels[3], self.M[3], self.in_channels[2], upsample=True)
        self.up_fblock3 = FBlock(self.in_channels[2], self.M[2], self.in_channels[1], upsample=True)
        self.up_fblock2 = FBlock(self.in_channels[1], self.M[1], self.in_channels[0], upsample=True)
        self.up_fblock1 = FBlock(self.in_channels[0], self.M[0], self.in_channels[0], upsample=False)


        self.convnext = nn.Sequential(
            BaseConv(3, self.in_channels[0], 3, 1, activation=nn.ReLU(inplace=True), use_bn=True),
            ConvNeXtV2_Block(self.in_channels[0]),
        )

        self.output_layer = nn.Sequential(
            BaseConv(2 * self.in_channels[0], self.in_channels[0], 1, 1, activation=nn.ReLU(inplace=True)),
            BaseConv(self.in_channels[0], 1, 3, 1),
        )

        self.mixconv4 = BaseConv1x1(self.in_channels[2] * 2, self.in_channels[2],)
        self.mixconv3 = BaseConv1x1(self.in_channels[1] * 2, self.in_channels[1],)
        self.mixconv2 = BaseConv1x1(self.in_channels[0] * 2, self.in_channels[0],)
        self.mixconv1 = BaseConv1x1(self.in_channels[0] * 2, self.in_channels[0],)

        self.mix_up4 = AT_UpSample(self.in_channels[2], self.in_channels[1])
        self.mix_up3 = AT_UpSample(self.in_channels[1], self.in_channels[0])
        # self.mix_up2 = AT_UpSample(self.in_channels[1], self.in_channels[0])


    def forward(self, x):
        convnext = self.convnext(x)

        conv_stem = self.stem_conv(x)

        conv1 = self.stage1(conv_stem)
        conv2 = self.stage2(self.down2(conv1))
        conv3 = self.stage3(self.down3(conv2))
        conv4 = self.stage4(self.down4(conv3))

        mscm4 = self.mscm4(conv4)
        mscm4_up = self.up4(mscm4)

        mscm3 = self.mscm3(conv3)
        de3 = self.de3(mscm3 + mscm4_up)
        mscm3_up = self.up3(de3)

        mscm2 = self.mscm2(conv2)
        de2 = self.de2(mscm2 + mscm3_up)
        mscm2_up = self.up2(de2)

        mscm1 = self.mscm1(conv1)
        de1 = self.de1(mscm1 + mscm2_up)

        pddp1 = self.pddp1(conv_stem)
        pddp2 = self.pddp2(self.down_lwga2(self.down_pool2(pddp1)))
        pddp3 = self.pddp3(self.down_lwga3(self.down_pool3(pddp2)))
        pddp4 = self.pddp4(self.down_lwga4(self.down_pool4(pddp3)))

        f4 = self.up_fblock4(pddp4)
        f4_c = f4 + pddp3
        f3 = self.up_fblock3(f4_c)
        f3_c = f3 + pddp2
        f2 = self.up_fblock2(f3_c)
        f2_c = f2 + pddp1
        f1 = self.up_fblock1(f2_c)

        mix4 = torch.cat((f4, mscm4_up), 1)
        mix4 = self.mixconv4(mix4)
        mix3 = torch.cat((f3, mscm3_up), 1)
        mix3 = self.mixconv3(mix3)
        mix2 = torch.cat((f2, mscm2_up), 1)
        mix2 = self.mixconv2(mix2)
        mix1 = torch.cat((f1, mscm1), 1)
        mix1 = self.mixconv1(mix1)

        mix4_up = self.mix_up4(mix4)
        mix3_up = self.mix_up3(mix3 + mix4_up)
        # mix2_up = self.mix_up2(mix2 + mix3_up)
        mix2_up = mix2 + mix3_up
        mix1_up = mix1 + mix2_up

        o = torch.cat((mix1_up, convnext), dim=1)

        o = self.output_layer(o)
        return torch.sigmoid(o)

    def _make_layer(self, block, dim, block_nums):
        layers = []
        for i in range(0, block_nums):
            layers.append(block(dim))
        return nn.Sequential(*layers)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixNet(16).to(device)
    x = torch.randn(1, 3, 480, 320).to(device)
    y = model(x)
    print(y.shape)  # 应为 [1, 1, 480, 320]
