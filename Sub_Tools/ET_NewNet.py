import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from edge import Conv2d, EdgeConv, CPDCBlock, PlainBlock
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
from Sub_Tools.ET_PDDP_PDBlock import PDDP_DiffBlock as PD_PDDPBlock

from Sub_Tools.ET_PDDPBlock import PDDPBlock as ET_PDDPBlock

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
            ET_PDDPBlock(M, M // 2, M),
            ET_PDDPBlock(M, M // 2, M),
            ET_PDDPBlock(M, M // 2, M),
            ET_PDDPBlock(M, M // 2, M),
            # PD_PDDPBlock(M, M // 2, M),
            # PD_PDDPBlock(M, M // 2, M),
            # PD_PDDPBlock(M, M // 2, M),
            # PD_PDDPBlock(M, M // 2, M),

        )
        self.poolblock1 = PoolBlock(M, M * 2, kernel_size=2, stride=2)
        self.PDDPList2 = nn.Sequential(
            ET_PDDPBlock(M * 2, M, M * 2),
            ET_PDDPBlock(M * 2, M, M * 2),
            ET_PDDPBlock(M * 2, M, M * 2),
            ET_PDDPBlock(M * 2, M, M * 2),
            # PD_PDDPBlock(M * 2, M, M * 2),
            # PD_PDDPBlock(M * 2, M, M * 2),
            # PD_PDDPBlock(M * 2, M, M * 2),
            # PD_PDDPBlock(M * 2, M, M * 2),
        )
        self.poolblock2 = PoolBlock(M * 2, M * 4, kernel_size=2, stride=2)
        self.PDDPList3 = nn.Sequential(
            ET_PDDPBlock(M * 4, M * 2, M * 4),
            ET_PDDPBlock(M * 4, M * 2, M * 4),
            ET_PDDPBlock(M * 4, M * 2, M * 4),
            ET_PDDPBlock(M * 4, M * 2, M * 4),
            # PD_PDDPBlock(M * 4, M * 2, M * 4),
            # PD_PDDPBlock(M * 4, M * 2, M * 4),
            # PD_PDDPBlock(M * 4, M * 2, M * 4),
            # PD_PDDPBlock(M * 4, M * 2, M * 4),
        )

        self.fblock1 = FBlock(M * 4, M * 2, M * 2)
        self.fblock2 = FBlock(M * 2, M, M)
        self.fblock3 = FBlock(M, M // 2, 1, upsample=False)

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

        return torch.sigmoid(f3)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ET_UHNet().to(device)
    x = torch.randn(1, 3, 480, 320).to(device)
    y = model(x)
    print(y.shape)  # 应为 [1, 1, 480, 320]
