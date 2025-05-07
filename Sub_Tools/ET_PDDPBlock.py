import torch
import torch.nn as nn
import torch.nn.functional as F
# from AT_UpSample2 import DySampleFusion_WithOut


class PDDPBlock(nn.Module):
    def __init__(self, C_in, M=-1, C_out=-1):
        super(PDDPBlock, self).__init__()
        if (M == -1):
            M = C_in // 2
        if (C_out == -1):
            C_out = C_in
        self.conv1x1_1 = nn.Conv2d(C_in, M, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M)
        self.relu1 = nn.ReLU(inplace=True)

        self.Dconv1 = nn.Conv2d(M, M, kernel_size=3, stride=1, padding=1, groups=M, bias=False)
        self.Dconv2 = nn.Conv2d(M, M, kernel_size=3, stride=1, padding=1, groups=M, bias=False)

        self.bn2 = nn.BatchNorm2d(M)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv1x1_2 = nn.Conv2d(M, C_out, kernel_size=1, bias=False)

        self.shortcut = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
    def forward(self, x):
        residual = x

        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.Dconv1(x)
        x = self.Dconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv1x1_2(x)

        shortcut = self.shortcut(residual)

        x = x + shortcut

        return x

class FBlock(nn.Module):
    def __init__(self, C_in, M=-1, C_out=-1):
        super(FBlock, self).__init__()
        if (M == -1):
            M = C_in // 2
        if (C_out == -1):
            C_out = C_in
        self.conv1x1_1 = nn.Conv2d(C_in, M, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M)
        self.relu1 = nn.ReLU(inplace=True)

        self.Dconv1 = nn.Conv2d(M, M, kernel_size=3, stride=1, padding=1, groups=M, bias=False)
        self.Dconv2 = nn.Conv2d(M, M, kernel_size=3, stride=1, padding=1, groups=M, bias=False)

        self.bn2 = nn.BatchNorm2d(M)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv1x1_2 = nn.Conv2d(M, C_out, kernel_size=1, bias=False)

    def forward(self, x):

        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.Dconv1(x)
        x = self.Dconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv1x1_2(x)

        return x

class PoolBlock(nn.Module):
    def __init__(self, C_in, C_out):
        super(PoolBlock, self).__init__()
        assert C_in == C_out or C_in *2 == C_out , "Error: C_in must be equal to C_out or C_in*2 == C_out"
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        if C_in == C_out:
            self.double = False
        else:
            self.double = True
    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        if self.double:
            return torch.cat((x1, x2), 1)
        else:
            return x1 + x2

# class Upsampling(nn.Module):
#     def __init__(self, C_in, C_out):
#         super(Upsampling, self).__init__()
#         self.up = DySampleFusion_WithOut(C_in, C_out,)
#     def forward(self, x):
#         return self.up(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
block = PDDPBlock(C_in=64, M=32, C_out=128).to(device)
x = torch.randn(1, 64, 480, 320).to(device)
y = block(x)
print(y.shape)  # 应为 [1, 128, 128, 128]