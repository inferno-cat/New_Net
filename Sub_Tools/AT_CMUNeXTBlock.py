import torch
import torch.nn as nn
import torch.nn.functional as F

# class MultiScaleConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], reduction=4):
#         super(MultiScaleConvBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_sizes = kernel_sizes
#         self.reduction = reduction
#
#         # Define depthwise convolutions with different kernel sizes
#         self.convs = nn.ModuleList([
#             nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=1, padding=k // 2, groups=in_channels)
#             for k in kernel_sizes
#         ])
#
#         # Pointwise convolution to combine channels
#         self.pointwise = nn.Conv2d(len(kernel_sizes) * in_channels, out_channels, kernel_size=1)
#
#         # Inverted bottleneck design to expand and then reduce channels
#         self.expand = nn.Conv2d(out_channels, out_channels * reduction, kernel_size=1)
#         self.reduce = nn.Conv2d(out_channels * reduction, out_channels, kernel_size=1)
#
#         # Batch normalization and activation
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.gelu = nn.GELU()
#
#     def forward(self, x):
#         # Apply each depthwise convolution and concatenate results along channel dimension
#         multi_scale_features = [conv(x) for conv in self.convs]
#         x = torch.cat(multi_scale_features, dim=1)
#
#         # Apply pointwise convolution to mix channels
#         x = self.pointwise(x)
#
#         # Inverted bottleneck: expand and then reduce channels
#         x = self.gelu(self.expand(x))
#         x = self.reduce(x)
#
#         # Apply batch normalization and GELU activation
#         x = self.bn(x)
#         x = self.gelu(x)
#
#         return x

import torch
import torch.nn as nn

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4], reduction=4):
        super(MultiScaleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.reduction = reduction

        # Define depthwise dilated convolutions with different dilation rates
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=dilation, dilation=dilation, groups=in_channels)
            for dilation in dilations
        ])

        # Pointwise convolution to combine channels
        self.pointwise = nn.Conv2d(len(dilations) * in_channels, out_channels, kernel_size=1)

        # Inverted bottleneck design to expand and then reduce channels
        self.expand = nn.Conv2d(out_channels, out_channels * reduction, kernel_size=1)
        self.reduce = nn.Conv2d(out_channels * reduction, out_channels, kernel_size=1)

        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Apply each dilated depthwise convolution and concatenate results along channel dimension
        multi_scale_features = [conv(x) for conv in self.convs]
        x = torch.cat(multi_scale_features, dim=1)

        # Apply pointwise convolution to mix channels
        x = self.pointwise(x)

        # Inverted bottleneck: expand and then reduce channels
        x = self.gelu(self.expand(x))
        x = self.reduce(x)

        # Apply batch normalization and GELU activation
        x = self.bn(x)
        x = self.gelu(x)

        return x

# Test the MultiScaleConvBlock
if __name__ == "__main__":
    # Example input tensor (batch_size=1, channels=32, height=64, width=64)
    input_tensor = torch.randn(3, 32, 64, 64)
    multi_scale_conv_block = MultiScaleConvBlock(in_channels=32, out_channels=32)
    output = multi_scale_conv_block(input_tensor)
    print('input_size:', input_tensor.shape)
    print("Output shape:", output.shape)
