import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from Sub_Tools.ET_PDDPBlock import PDDPBlock
# from Sub_Tools.AT_CBAM import EAT_CBAM, CBAM
# from Sub_Tools.AT_BIFormer_dcart import BiLevelRoutingAttention_nchw as ET_attn
from Sub_Tools.AT_MSPA import MSPAModule as ET_attn
from Sub_Tools.AT_RepVitBlock import GlobalAttentionRepViTBlock as AT_Rep

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

class MultiScaleSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(MultiScaleSE, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d(2)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局和局部池化
        global_se = self.global_pool(x)
        local_se = self.local_pool(x)
        # 上采样局部池化结果并叠加
        se = global_se + nn.functional.interpolate(local_se, size=global_se.shape[-2:])
        se = self.fc1(se)
        se = nn.ReLU()(se)
        se = self.sigmoid(self.fc2(se))
        return x * se

class CPDCBlock(nn.Module):
    def __init__(self, in_channels):
        super(CPDCBlock, self).__init__()
        self.conv_d = nn.Sequential(
            ConvFac(Conv2d_d, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )
        self.conv_v = nn.Sequential(
            ConvFac(Conv2d_v, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )
        self.conv_h = nn.Sequential(
            ConvFac(Conv2d_h, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )
        self.conv_c = nn.Sequential(
            ConvFac(Conv2d_c, in_channels, in_channels // 4, 3, 1, 1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(True),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        # self.conv_rep = AT_Rep(in_channels, in_channels, kernel_size=3, stride=1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        # self.attn = CBAM(in_channels, reduction_ratio=16,)
        # self.attn = MultiScaleSE(in_channels)
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
        residual = x

        x_d = self.conv_d(x)
        x_v = self.conv_v(x)
        x_h = self.conv_h(x)
        x_c = self.conv_c(x)

        x = torch.cat([x_d, x_v, x_h, x_c], dim=1)
        x = self.conv3x3(x)
        # x = self.conv_rep(x)
        # x = self.attn(x) + x

        x = self.conv1x1(x)
        # x = self.attn(x) + x

        x = x + residual

        return x

    def convert_to_standard_conv(self):
        for module in self.modules():
            if isinstance(module, ConvFac):
                module.convert_to_standard_conv()

class MixBlock(nn.Module):
    def __init__(self, in_channels, ):
        super(MixBlock, self).__init__()
        self.cpdc_block = CPDCBlock(in_channels)
        # self.attn_block = ET_attn(inplanes=in_channels // 4, scale=4)
        self.attn_block = AT_Rep(in_channels, in_channels, kernel_size=3, stride=1)
        self.mixconv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        residual = x
        cpdc = self.cpdc_block(x)
        attn = self.attn_block(x)

        o = torch.cat([cpdc, attn], dim=1)
        o = self.mixconv(o) + residual
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
