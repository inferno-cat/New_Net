import torch
import torch.nn as nn
import torch.nn.functional as F

# 垂直方向深度卷积
class DepthConv2d_v(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConv2d_v, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_v = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_v = conv_weight_v - conv_weight_v[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]]
        conv_weight_v = conv_weight_v.view(conv_shape)
        return conv_weight_v, self.conv.bias

# 水平方向深度卷积
class DepthConv2d_h(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConv2d_h, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_h = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_h = conv_weight_h - conv_weight_h[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]]
        conv_weight_h = conv_weight_h.view(conv_shape)
        return conv_weight_h, self.conv.bias

# 中心方向深度卷积
class DepthConv2d_c(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConv2d_c, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_c = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_c[:, :, [1, 3, 5, 7]] = conv_weight_c[:, :, [1, 3, 5, 7]] - conv_weight_c[:, :, [7, 5, 4, 4]]
        conv_weight_c[:, :, [4]] = 2 * conv_weight_c[:, :, [4]] - conv_weight_c[:, :, [3]] - conv_weight_c[:, :, [1]]
        conv_weight_c[:, :, [0, 2, 6, 8]] = 0
        conv_weight_c = conv_weight_c.view(conv_shape)
        return conv_weight_c, self.conv.bias

# 对角线方向深度卷积
class DepthConv2d_d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConv2d_d, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_d = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        conv_weight_d[:, :, [0, 2, 4, 6]] = conv_weight_d[:, :, [0, 2, 4, 6]] - conv_weight_d[:, :, [8, 6, 4, 4]]
        conv_weight_d[:, :, [4]] = 2 * conv_weight_d[:, :, [4]] - conv_weight_d[:, :, [0]] - conv_weight_d[:, :, [2]]
        conv_weight_d[:, :, [1, 3, 5, 7]] = 0
        conv_weight_d = conv_weight_d.view(conv_shape)
        return conv_weight_d, self.conv.bias

# 包装器，支持动态权重和标准卷积切换
class DepthConvFac(nn.Module):
    def __init__(self, op_type, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConvFac, self).__init__()
        assert op_type in [DepthConv2d_v, DepthConv2d_h, DepthConv2d_c, DepthConv2d_d], "Invalid op_type"
        self.op_type = op_type(channels, kernel_size, stride, padding, groups, bias)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.channels = channels
        self.standard_conv = None

    def forward(self, x):
        if self.standard_conv is None:
            w, b = self.op_type.get_weight()
            res = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            res = self.standard_conv(x)
        return res

    def convert_to_standard_conv(self):
        if self.standard_conv is None:
            w, b = self.op_type.get_weight()
            self.standard_conv = nn.Conv2d(
                self.channels, self.channels, 3, stride=self.stride, padding=self.padding,
                groups=self.groups, bias=b is not None
            )
            self.standard_conv.weight.data = w
            if b is not None:
                self.standard_conv.bias.data = b

# 新模块：PDDP_FourDiffBlock
class PDDP_FourDiffBlock(nn.Module):
    def __init__(self, C_in, M, C_out):
        super(PDDP_FourDiffBlock, self).__init__()
        assert C_in > 0 and M > 0 and C_out > 0, "Channel numbers must be positive"
        assert M % 4 == 0, "M must be divisible by 4 for four branches"

        # 1x1 卷积降维
        self.conv1x1_1 = nn.Conv2d(C_in, M, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M)
        self.relu1 = nn.ReLU(inplace=True)
        self.M = M
        # 四分支，每分支两层方向性深度卷积
        M_branch = M // 4  # 每个分支的通道数
        # 垂直分支
        self.branch_v = nn.Sequential(
            DepthConvFac(DepthConv2d_v, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False),
            DepthConvFac(DepthConv2d_v, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False)
        )
        # 水平分支
        self.branch_h = nn.Sequential(
            DepthConvFac(DepthConv2d_h, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False),
            DepthConvFac(DepthConv2d_h, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False)
        )
        # 中心分支
        self.branch_c = nn.Sequential(
            DepthConvFac(DepthConv2d_c, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False),
            DepthConvFac(DepthConv2d_c, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False)
        )
        # 对角线分支
        self.branch_d = nn.Sequential(
            DepthConvFac(DepthConv2d_d, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False),
            DepthConvFac(DepthConv2d_d, M_branch, kernel_size=3, stride=1, padding=1, groups=M_branch, bias=False)
        )

        # 批归一化和激活
        self.bn2 = nn.BatchNorm2d(M)
        self.relu2 = nn.ReLU(inplace=True)

        # 1x1 卷积升维
        self.conv1x1_2 = nn.Conv2d(M, C_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(C_out)
        self.relu3 = nn.ReLU(inplace=True)

        # 残差连接
        self.shortcut = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(C_out)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        # 主路径
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 四分支并行处理
        M_branch = self.M // 4
        x_v = self.branch_v(x[:, :M_branch])
        x_h = self.branch_h(x[:, M_branch:2*M_branch])
        x_c = self.branch_c(x[:, 2*M_branch:3*M_branch])
        x_d = self.branch_d(x[:, 3*M_branch:])
        x = torch.cat([x_v, x_h, x_c, x_d], dim=1)  # 拼接恢复 M 通道

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 残差路径
        shortcut = self.shortcut_bn(self.shortcut(residual))

        # 残差连接
        x = x + shortcut
        return x

    def convert_to_standard_conv(self):
        """将所有方向性卷积转换为标准卷积以优化推理"""
        for branch in [self.branch_v, self.branch_h, self.branch_c, self.branch_d]:
            for layer in branch:
                if isinstance(layer, DepthConvFac):
                    layer.convert_to_standard_conv()

# 测试用例
if __name__ == "__main__":
    M = 32  # 确保 M 可被 4 整除
    block = PDDP_FourDiffBlock(C_in=64, M=M, C_out=128)
    x = torch.randn(1, 64, 128, 128)
    y = block(x)
    print(f"Output shape: {y.shape}")  # 应为 [1, 128, 128, 128]
    block.convert_to_standard_conv()
    y = block(x)
    print(f"Output shape after conversion: {y.shape}")



import torch
import torch.nn as nn
import torch.nn.functional as F

# 垂直方向深度卷积（基于 Conv2d_v 的像素差分思想）
class DepthConv2d_v(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConv2d_v, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_v = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        # 垂直方向像素差分
        conv_weight_v = conv_weight_v - conv_weight_v[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]]
        conv_weight_v = conv_weight_v.view(conv_shape)
        return conv_weight_v, self.conv.bias

# 水平方向深度卷积（基于 Conv2d_h 的像素差分思想）
class DepthConv2d_h(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConv2d_h, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_h = conv_weight.view(conv_shape[0], conv_shape[1], -1).clone()
        # 水平方向像素差分
        conv_weight_h = conv_weight_h - conv_weight_h[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]]
        conv_weight_h = conv_weight_h.view(conv_shape)
        return conv_weight_h, self.conv.bias

# 包装器，用于支持动态权重和标准卷积切换
class DepthConvFac(nn.Module):
    def __init__(self, op_type, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(DepthConvFac, self).__init__()
        assert op_type in [DepthConv2d_v, DepthConv2d_h], "op_type must be DepthConv2d_v or DepthConv2d_h"
        self.op_type = op_type(channels, kernel_size, stride, padding, groups, bias)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.channels = channels
        self.standard_conv = None

    def forward(self, x):
        if self.standard_conv is None:
            w, b = self.op_type.get_weight()
            res = F.conv2d(x, weight=w, bias=b, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            res = self.standard_conv(x)
        return res

    def convert_to_standard_conv(self):
        if self.standard_conv is None:
            w, b = self.op_type.get_weight()
            self.standard_conv = nn.Conv2d(
                self.channels, self.channels, 3, stride=self.stride, padding=self.padding,
                groups=self.groups, bias=b is not None
            )
            self.standard_conv.weight.data = w
            if b is not None:
                self.standard_conv.bias.data = b

# 新模块：PDDP-DiffBlock
class PDDP_DiffBlock(nn.Module):
    def __init__(self, C_in, M, C_out):
        super(PDDP_DiffBlock, self).__init__()
        assert C_in > 0 and M > 0 and C_out > 0, "Channel numbers must be positive"

        # 1x1 卷积降维
        self.conv1x1_1 = nn.Conv2d(C_in, M, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(M)
        self.relu1 = nn.ReLU(inplace=True)

        # 两个方向性深度卷积（垂直和水平）
        self.Dconv1 = DepthConvFac(DepthConv2d_v, M, kernel_size=3, stride=1, padding=1, groups=M, bias=False)
        self.Dconv2 = DepthConvFac(DepthConv2d_h, M, kernel_size=3, stride=1, padding=1, groups=M, bias=False)
        self.bn2 = nn.BatchNorm2d(M)
        self.relu2 = nn.ReLU(inplace=True)

        # 1x1 卷积升维
        self.conv1x1_2 = nn.Conv2d(M, C_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(C_out)
        self.relu3 = nn.ReLU(inplace=True)

        # 残差连接
        self.shortcut = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(C_out)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        # 主路径
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.Dconv1(x)
        x = self.Dconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 残差路径
        shortcut = self.shortcut_bn(self.shortcut(residual))

        # 残差连接
        x = x + shortcut

        return x

    def convert_to_standard_conv(self):
        """将所有方向性卷积转换为标准卷积以优化推理"""
        self.Dconv1.convert_to_standard_conv()
        self.Dconv2.convert_to_standard_conv()

# 测试用例
if __name__ == "__main__":
    block = PDDP_DiffBlock(C_in=64, M=32, C_out=128)
    x = torch.randn(1, 64, 128, 128)
    y = block(x)
    print(f"Output shape: {y.shape}")  # 应为 [1, 128, 128, 128]
    block.convert_to_standard_conv()
    y = block(x)
    print(f"Output shape after conversion: {y.shape}")