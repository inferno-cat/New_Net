import torch
import numpy as np
import math
import torch.nn as nn
import pywt
from torch.autograd import Function

class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        batch, channels, height, width = input.size()
        input_2d = input.view(batch * channels, height, width)
        L = torch.matmul(matrix_Low_0, input_2d)  # 行方向低通滤波: [batch*channels, L_h, width]
        H = torch.matmul(matrix_High_0, input_2d)  # 行方向高通滤波
        LL = torch.matmul(L, matrix_Low_1)  # 列方向低通滤波: [batch*channels, L_h, L_w]
        LH = torch.matmul(L, matrix_High_1)  # 列方向高通滤波
        HL = torch.matmul(H, matrix_Low_1)  # 列方向低通滤波
        HH = torch.matmul(H, matrix_High_1)  # 列方向高通滤波
        LL = LL.view(batch, channels, LL.size(1), LL.size(2))
        LH = LH.view(batch, channels, LH.size(1), LH.size(2))
        HL = HL.view(batch, channels, HL.size(1), HL.size(2))
        HH = HH.view(batch, channels, HH.size(1), HH.size(2))
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()),
                           torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()),
                           torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L),
                               torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None

class DWT_2D(nn.Module):
    def __init__(self, wavename):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L_h = math.floor(self.input_height / 2)  # 高度下采样后的尺寸
        L_w = math.floor(self.input_width / 2)   # 宽度下采样后的尺寸

        # 初始化滤波矩阵
        matrix_h = np.zeros((L_h, self.input_height))  # 行方向低通滤波
        matrix_g = np.zeros((L_h, self.input_height))  # 行方向高通滤波
        matrix_h_1 = np.zeros((self.input_width, L_w))  # 列方向低通滤波
        matrix_g_1 = np.zeros((self.input_width, L_w))  # 列方向高通滤波

        # 填充行方向滤波矩阵
        for i in range(L_h):
            for j in range(self.band_length):
                idx = (2 * i + j) % self.input_height
                matrix_h[i, idx] = self.band_low[j]
                matrix_g[i, idx] = self.band_high[j]

        # 填充列方向滤波矩阵
        for i in range(L_w):
            for j in range(self.band_length):
                idx = (2 * i + j) % self.input_width
                matrix_h_1[idx, i] = self.band_low[j]
                matrix_g_1[idx, i] = self.band_high[j]

        # 转换为张量并移动到输入设备
        device = self.input.device
        self.matrix_low_0 = torch.Tensor(matrix_h).to(device)
        self.matrix_low_1 = torch.Tensor(matrix_h_1).to(device)
        self.matrix_high_0 = torch.Tensor(matrix_g).to(device)
        self.matrix_high_1 = torch.Tensor(matrix_g_1).to(device)

    def forward(self, input):
        assert len(input.size()) == 4
        self.input = input
        self.input_height = self.input.size()[-2]
        self.input_width = self.input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(self.input, self.matrix_low_0, self.matrix_low_1,
                                    self.matrix_high_0, self.matrix_high_1)

class WPL(nn.Module):
    def __init__(self, wavename='haar'):
        super(WPL, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)
        output = LL
        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        return output

class WPL_Outchannels(nn.Module):
    def __init__(self, in_channels, out_channels, wavename='haar'):
        super(WPL_Outchannels, self).__init__()
        self.wpl = WPL(wavename=wavename)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        x = self.wpl(input)
        x = self.conv(x)
        return x

if __name__ == "__main__":
    batch_size = 1
    in_channels = 16
    out_channels = 64
    height, width = 480, 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)

    wpl = WPL(wavename='haar').to(device)
    output_wpl = wpl(input_tensor)
    print(f"WPL Input shape: {input_tensor.shape}")
    print(f"WPL Output shape: {output_wpl.shape}")

    wpl_outchannels = WPL_Outchannels(
        in_channels=in_channels,
        out_channels=out_channels,
        wavename='haar'
    ).to(device)
    output_wpl_out = wpl_outchannels(input_tensor)
    print(f"WPL_Outchannels Input shape: {input_tensor.shape}")
    print(f"WPL_Outchannels Output shape: {output_wpl_out.shape}")


import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
'''
CV缝合救星魔改创新：WTFD下采样模块

思路：小波变化的特点是将高频和低频分别进行提取，提取后特征图W,H减半，和下采样的特性类似。
实现：将低频和高频特征进行缝合。
'''
class WTFDown(nn.Module):#小波变化高低频分解下采样模块
    def __init__(self, in_ch, out_ch):
        super(WTFDown, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        yL_gate = self.sigmod(yL)
        return  yH * yL_gate


if __name__ == "__main__":
    # 创建一个简单的输入特征图
    input = torch.randn(1, 32, 480, 320)
    testWTFDown = WTFDown(32,64)  #小波变化高低频分解下采样模块
    output = testWTFDown(input)
    print(f"input  shape: {input.shape}")
    print(f"output shape: {output.shape}")



