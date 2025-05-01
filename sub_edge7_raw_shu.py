import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from Sub_Tools.ET_PDDPBlock import PDDPBlock
# from Sub_Tools.AT_CBAM import EAT_CBAM, CBAM
# from Sub_Tools.AT_BIFormer_dcart import BiLevelRoutingAttention_nchw as ET_attn
# from Sub_Tools.AT_MSPA import MSPAModule as ET_attn
# from Sub_Tools.AT_EMA import EMA as ET_attn
# from Sub_Tools.AT_DA import MultiDilatelocalAttention as ET_attn
from Sub_Tools.XT_CoordAttention import CoordAttention as ET_attn

class Conv2d(nn.Module):
    def __init__(
        self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False
    ):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        # self.pdc_func = createConvFunc(pdc_func)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2d_v(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_v, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_v = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_v = conv_weight_v - conv_weight_v[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]]
        conv_weight_v = conv_weight_v.view(conv_shape)
        return conv_weight_v, self.conv.bias

class Conv2d_h(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_h, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_h = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_h = conv_weight_h - conv_weight_h[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]]
        conv_weight_h = conv_weight_h.view(conv_shape)
        return conv_weight_h, self.conv.bias

class Conv2d_c(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_c, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_c = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_c[:, :, [1, 3, 5, 7]] = conv_weight_c[:, :, [1, 3, 5, 7]] - conv_weight_c[:, :, [7, 5, 4, 4]]
        conv_weight_c[:, :, [4]] = 2 * conv_weight_c[:, :, [4]] - conv_weight_c[:, :, [3]] - conv_weight_c[:, :, [1]]
        conv_weight_c[:, :, [0, 2, 6, 8]] = 0
        conv_weight_c = conv_weight_c.view(conv_shape)
        return conv_weight_c, self.conv.bias

class Conv2d_d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_d = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_d[:, :, [0, 2, 4, 6]] = conv_weight_d[:, :, [0, 2, 4, 6]] - conv_weight_d[:, :, [8, 6, 4, 4]]
        conv_weight_d[:, :, [4]] = 2 * conv_weight_d[:, :, [4]] - conv_weight_d[:, :, [0]] - conv_weight_d[:, :, [2]]
        conv_weight_d[:, :, [1, 3, 5, 7]] = 0
        conv_weight_d = conv_weight_d.view(conv_shape)
        return conv_weight_d, self.conv.bias

class ConvFac(nn.Module):
    def __init__(
        self, op_type, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False
    ):
        super(ConvFac, self).__init__()
        assert op_type in [Conv2d_v, Conv2d_h, Conv2d_c, Conv2d_d]#, Conv2d_cd, Conv2d_ad, Conv2d_rd]
        self.op_type = op_type(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.standard_conv = None

    def forward(self, x):
        if self.standard_conv is None:
            w, b = self.op_type.get_weight()
            res = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            res = self.standard_conv(x)

        return res

    def convert_to_standard_conv(self):
        w, b = self.op_type.get_weight()
        self.standard_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            w.size(2),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=b is not None,
        )
        self.standard_conv.weight.data = w
        if b is not None:
            self.standard_conv.bias.data = b

    def load_state_dict(self, state_dict, strict=True):
        if 'standard_conv.weight' in state_dict:
            if self.standard_conv is None:
                self.convert_to_standard_conv()
            self.standard_conv.load_state_dict(state_dict, strict)
        else:
            super().load_state_dict(state_dict, strict)


import torch
import torch.nn as nn
import math


def channel_shuffle(x, groups):
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, -1, height, width)
    return x

class CPDCBlock(nn.Module):
    def __init__(self, in_channels, groups_init=4):
        super(CPDCBlock, self).__init__()
        assert in_channels % 8 == 0, "in_channels must be divisible by 8"

        # 初始 1x1 分组卷积
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, groups=groups_init, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

        # 分支通道数
        branch_channels = in_channels // 8  # 每个分支的输入/输出通道数
        branch_groups = branch_channels  # 每个分支的分组数（深度卷积）

        # 四个分支（conv_d, conv_v, conv_h, conv_c）
        self.conv_d = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1, groups=branch_groups, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1, groups=branch_groups, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1, groups=branch_groups, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1, groups=branch_groups, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )

        # 合并后的 1x1 分组卷积
        self.conv_merge = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 1, 1, 0, groups=groups_init, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(True)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        # 初始 1x1 分组卷积 + 通道混洗
        x = self.conv_init(x)
        x = channel_shuffle(x, groups=4)

        # 通道分割：C/2 作为残差，C/2 分为四份
        residual, x = x.chunk(2, dim=1)  # residual: C/2, x: C/2
        x1, x2, x3, x4 = x.chunk(4, dim=1)  # 每个分支：C/8

        # 四个分支处理
        x_d = self.conv_d(x1)
        x_v = self.conv_v(x2)
        x_h = self.conv_h(x3)
        x_c = self.conv_c(x4)

        # 合并分支
        x = torch.cat([x_d, x_v, x_h, x_c], dim=1)  # 输出：C/2

        # 合并后的 1x1 分组卷积 + 通道混洗
        x = self.conv_merge(x)
        x = channel_shuffle(x, groups=4)

        # 与残差分支合并
        x = torch.cat([x, residual], dim=1)  # 输出：C

        return x

class MixBlock(nn.Module):
    def __init__(self, in_channels, ):
        super(MixBlock, self).__init__()
        self.cpdc_block = CPDCBlock(in_channels)
        self.attn_block = ET_attn(in_channels)
        self.mixconv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        cpdc = self.cpdc_block(x)
        attn = self.attn_block(x)
        o = torch.cat([cpdc, attn], dim=1)
        o = self.mixconv(o)
        return o

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super(EdgeConv, self).__init__()
        self.bias = bias
        self.groups = groups
        self.conv_v = ConvFac(Conv2d_v, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_h = ConvFac(Conv2d_h, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_c = ConvFac(Conv2d_h, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_d = ConvFac(Conv2d_h, in_channels, out_channels, 3, groups=groups, bias=bias)
        self.conv_p = nn.Conv2d(in_channels, out_channels, 3, groups=groups, bias=bias)

    def forward(self, x):
        w_v, b_v = self.conv_v.get_weight()
        w_h, b_h = self.conv_h.get_weight()
        w_c, b_c = self.conv_c.get_weight()
        w_d, b_d = self.conv_d.get_weight()
        w_p, b_p = self.conv_p.weight, self.conv_p.bias

        w = w_v + w_h + w_c + w_d + w_p

        if self.bias:
            b = b_v + b_h + b_c + b_d + b_p
        else:
            b = None

        res = F.conv2d(x, weight=w, bias=b, stride=1, padding=1, groups=self.groups)

        return res


if __name__ == "__main__":
    # weights = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    # weights_conv = weights.clone()
    # theta = 1.0
    # weights_conv[[1, 3]] = weights[[1, 3]] - theta * weights[[4]]
    # weights_conv[[5, 7]] = weights[[4]] - theta * weights[[5, 7]]
    # weights_conv[[0, 2, 6, 8]] = 1 - theta
    # print(weights_conv.view(3, 3))
    conv = CPDCBlock(32)
    # conv = ConvFactor(Conv2d, 3, 3, bias=False)
    # print(conv)
    x = torch.randn(4, 32, 480, 320)
    out = conv(x)
    print(out.shape)
