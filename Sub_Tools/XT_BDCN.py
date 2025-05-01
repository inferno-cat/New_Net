import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Generate 2D bilinear kernel for upsampling."""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class MSBlock(nn.Module):
    """Scale Enhancement Module (SEM) with multi-scale dilated convolutions."""
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        self.rate = rate
        c_out = c_in  # Output channels match input channels for residual connection
        # Initial 3x3 convolution
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Dilated convolutions with different rates
        dilation = self.rate * 1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(c_out, c_out, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate * 2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate * 3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(c_out, c_out, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        print(f"MSBlock input shape: {x.shape}")
        o = self.relu(self.conv(x))
        print(f"MSBlock conv output shape: {o.shape}")
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        # Residual connection
        out = x + o + o1 + o2 + o3
        print(f"MSBlock output shape: {out.shape}")
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class BDCN(nn.Module):
    """Bi-Directional Cascade Network for edge detection."""
    def __init__(self, pretrain=True, rate=4):
        super(BDCN, self).__init__()
        self.rate = rate

        # Load pretrained VGG-16
        vgg16 = models.vgg16(weights='IMAGENET1K_V1' if pretrain else None)
        self.features = nn.Sequential(*list(vgg16.features)[:-1])  # Remove pool5

        # Channel configurations for VGG-16 blocks
        self.channel_configs = {
            'conv1': 64,  # conv1_1, conv1_2
            'conv2': 128,  # conv2_1, conv2_2
            'conv3': 256,  # conv3_1, conv3_2, conv3_3
            'conv4': 512,  # conv4_1, conv4_2, conv4_3
            'conv5': 512   # conv5_1, conv5_2, conv5_3
        }

        # Define SEM and side output layers for each block
        # Block 1: conv1_1, conv1_2 (64 channels)
        self.msblock1_1 = MSBlock(self.channel_configs['conv1'], rate)
        self.msblock1_2 = MSBlock(self.channel_configs['conv1'], rate)
        self.conv1_1_down = nn.Conv2d(self.channel_configs['conv1'], 21, 1, stride=1)
        self.conv1_2_down = nn.Conv2d(self.channel_configs['conv1'], 21, 1, stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, 1, stride=1)  # s2d
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)  # d2s

        # Block 2: conv2_1, conv2_2 (128 channels)
        self.msblock2_1 = MSBlock(self.channel_configs['conv2'], rate)
        self.msblock2_2 = MSBlock(self.channel_configs['conv2'], rate)
        self.conv2_1_down = nn.Conv2d(self.channel_configs['conv2'], 21, 1, stride=1)
        self.conv2_2_down = nn.Conv2d(self.channel_configs['conv2'], 21, 1, stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1, stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, 1, stride=1)

        # Block 3: conv3_1, conv3_2, conv3_3 (256 channels)
        self.msblock3_1 = MSBlock(self.channel_configs['conv3'], rate)
        self.msblock3_2 = MSBlock(self.channel_configs['conv3'], rate)
        self.msblock3_3 = MSBlock(self.channel_configs['conv3'], rate)
        self.conv3_1_down = nn.Conv2d(self.channel_configs['conv3'], 21, 1, stride=1)
        self.conv3_2_down = nn.Conv2d(self.channel_configs['conv3'], 21, 1, stride=1)
        self.conv3_3_down = nn.Conv2d(self.channel_configs['conv3'], 21, 1, stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1, stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, 1, stride=1)

        # Block 4: conv4_1, conv4_2, conv4_3 (512 channels)
        self.msblock4_1 = MSBlock(self.channel_configs['conv4'], rate)
        self.msblock4_2 = MSBlock(self.channel_configs['conv4'], rate)
        self.msblock4_3 = MSBlock(self.channel_configs['conv4'], rate)
        self.conv4_1_down = nn.Conv2d(self.channel_configs['conv4'], 21, 1, stride=1)
        self.conv4_2_down = nn.Conv2d(self.channel_configs['conv4'], 21, 1, stride=1)
        self.conv4_3_down = nn.Conv2d(self.channel_configs['conv4'], 21, 1, stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1, stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, 1, stride=1)

        # Block 5: conv5_1, conv5_2, conv5_3 (512 channels)
        self.msblock5_1 = MSBlock(self.channel_configs['conv5'], rate)
        self.msblock5_2 = MSBlock(self.channel_configs['conv5'], rate)
        self.msblock5_3 = MSBlock(self.channel_configs['conv5'], rate)
        self.conv5_1_down = nn.Conv2d(self.channel_configs['conv5'], 21, 1, stride=1)
        self.conv5_2_down = nn.Conv2d(self.channel_configs['conv5'], 21, 1, stride=1)
        self.conv5_3_down = nn.Conv2d(self.channel_configs['conv5'], 21, 1, stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1, stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, 1, stride=1)

        # Channel reduction for backward path
        self.reduce_channels_4 = nn.Conv2d(512, 256, 1, stride=1)  # conv5 to conv3
        self.reduce_channels_3 = nn.Conv2d(512, 128, 1, stride=1)  # conv4 to conv2
        self.reduce_channels_2 = nn.Conv2d(256, 64, 1, stride=1)   # conv3 to conv1

        # Fusion layer
        self.fuse = nn.Conv2d(10, 1, 1, stride=1)

        # Upsampling weights
        self.upsample_2 = nn.Conv2d(1, 1, 4, stride=2, padding=1)
        self.upsample_4 = nn.Conv2d(1, 1, 8, stride=4, padding=2)
        self.upsample_8 = nn.Conv2d(1, 1, 16, stride=8, padding=4)
        self.upsample_16 = nn.Conv2d(1, 1, 32, stride=16, padding=8)

        # Initialize upsampling weights
        for m in [self.upsample_2, self.upsample_4, self.upsample_8, self.upsample_16]:
            m.weight.data.copy_(get_upsampling_weight(1, 1, m.kernel_size[0]))
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.data.zero_()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m not in [self.upsample_2, self.upsample_4, self.upsample_8, self.upsample_16]:
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h, w = x.size()[2:]
        print(f"Input shape: {x.shape}")

        # Forward path: extract features from VGG-16
        features = []
        # conv1: conv1_1, conv1_2
        x1 = self.features[0:4](x)  # conv1_1, conv1_2 (64 channels)
        features.append(x1)
        print(f"conv1 output shape: {x1.shape}")
        # conv2: conv2_1, conv2_2
        x2 = self.features[4:9](x1)  # conv2_1, conv2_2 (128 channels)
        features.append(x2)
        print(f"conv2 output shape: {x2.shape}")
        # conv3: conv3_1, conv3_2, conv3_3
        x3 = self.features[9:16](x2)  # conv3_1, conv3_2, conv3_3 (256 channels)
        features.append(x3)
        print(f"conv3 output shape: {x3.shape}")
        # conv4: conv4_1, conv4_2, conv4_3
        x4 = self.features[16:23](x3)  # conv4_1, conv4_2, conv4_3 (512 channels)
        features.append(x4)
        print(f"conv4 output shape: {x4.shape}")
        # conv5: conv5_1, conv5_2, conv5_3
        x5 = self.features[23:](x4)  # conv5_1, conv5_2, conv5_3 (512 channels)
        features.append(x5)
        print(f"conv5 output shape: {x5.shape}")

        # Side outputs (shallow-to-deep, s2d)
        s1 = self.msblock1_2(self.msblock1_1(features[0]))
        print(f"s1 shape after MSBlock: {s1.shape}")
        s1 = self.conv1_2_down(s1)
        d1 = self.score_dsn1(s1)
        d1 = self.upsample_2(d1)[:, :, :h, :w]
        print(f"d1 (s2d) shape: {d1.shape}")

        s2 = self.msblock2_2(self.msblock2_1(features[1]))
        print(f"s2 shape after MSBlock: {s2.shape}")
        s2 = self.conv2_2_down(s2)
        d2 = self.score_dsn2(s2)
        d2 = self.upsample_4(d2)[:, :, :h, :w]
        print(f"d2 (s2d) shape: {d2.shape}")

        s3 = self.msblock3_3(self.msblock3_2(self.msblock3_1(features[2])))
        print(f"s3 shape after MSBlock: {s3.shape}")
        s3 = self.conv3_3_down(s3)
        d3 = self.score_dsn3(s3)
        d3 = self.upsample_8(d3)[:, :, :h, :w]
        print(f"d3 (s2d) shape: {d3.shape}")

        s4 = self.msblock4_3(self.msblock4_2(self.msblock4_1(features[3])))
        print(f"s4 shape after MSBlock: {s4.shape}")
        s4 = self.conv4_3_down(s4)
        d4 = self.score_dsn4(s4)
        d4 = self.upsample_16(d4)[:, :, :h, :w]
        print(f"d4 (s2d) shape: {d4.shape}")

        s5 = self.msblock5_3(self.msblock5_2(self.msblock5_1(features[4])))
        print(f"s5 shape after MSBlock: {s5.shape}")
        s5 = self.conv5_3_down(s5)
        d5 = self.score_dsn5(s5)
        d5 = self.upsample_16(d5)[:, :, :h, :w]
        print(f"d5 (s2d) shape: {d5.shape}")

        # Backward path (deep-to-shallow, d2s)
        s5_b = self.conv5_3_down(self.msblock5_3(self.msblock5_2(self.msblock5_1(features[4]))))
        print(f"s5_b shape after MSBlock: {s5_b.shape}")
        d5_b = self.score_dsn5_1(s5_b)
        d5_b = self.upsample_16(d5_b)[:, :, :h, :w]
        print(f"d5_b (d2s) shape: {d5_b.shape}")

        # conv5 to conv4: reduce channels from 512 to 512 (no reduction needed)
        s4_b = self.msblock4_3(self.msblock4_2(self.msblock4_1(F.interpolate(features[4], scale_factor=2))))
        print(f"s4_b shape after MSBlock: {s4_b.shape}")
        s4_b = self.conv4_3_down(s4_b)
        d4_b = self.score_dsn4_1(s4_b)
        d4_b = self.upsample_8(d4_b)[:, :, :h, :w]
        print(f"d4_b (d2s) shape: {d4_b.shape}")

        # conv5 to conv3: reduce channels from 512 to 256
        s3_b_input = self.reduce_channels_4(F.interpolate(features[4], scale_factor=4))
        print(f"s3_b input shape after channel reduction: {s3_b_input.shape}")
        s3_b = self.msblock3_3(self.msblock3_2(self.msblock3_1(s3_b_input)))
        print(f"s3_b shape after MSBlock: {s3_b.shape}")
        s3_b = self.conv3_3_down(s3_b)
        d3_b = self.score_dsn3_1(s3_b)
        d3_b = self.upsample_4(d3_b)[:, :, :h, :w]
        print(f"d3_b (d2s) shape: {d3_b.shape}")

        # conv4 to conv2: reduce channels from 512 to 128
        s2_b_input = self.reduce_channels_3(F.interpolate(features[3], scale_factor=8))
        print(f"s2_b input shape after channel reduction: {s2_b_input.shape}")
        s2_b = self.msblock2_2(self.msblock2_1(s2_b_input))
        print(f"s2_b shape after MSBlock: {s2_b.shape}")
        s2_b = self.conv2_2_down(s2_b)
        d2_b = self.score_dsn2_1(s2_b)
        d2_b = self.upsample_2(d2_b)[:, :, :h, :w]
        print(f"d2_b (d2s) shape: {d2_b.shape}")

        # conv3 to conv1: reduce channels from 256 to 64
        s1_b_input = self.reduce_channels_2(F.interpolate(features[2], scale_factor=16))
        print(f"s1_b input shape after channel reduction: {s1_b_input.shape}")
        s1_b = self.msblock1_2(self.msblock1_1(s1_b_input))
        print(f"s1_b shape after MSBlock: {s1_b.shape}")
        s1_b = self.conv1_2_down(s1_b)
        d1_b = self.score_dsn1_1(s1_b)
        d1_b = d1_b[:, :, :h, :w]
        print(f"d1_b (d2s) shape: {d1_b.shape}")

        # Fusion layer
        fuse = torch.cat([d1, d2, d3, d4, d5, d1_b, d2_b, d3_b, d4_b, d5_b], dim=1)
        print(f"Fuse input shape: {fuse.shape}")
        fuse = self.fuse(fuse)
        print(f"Fuse output shape: {fuse.shape}")

        return [fuse, d1, d2, d3, d4, d5, d1_b, d2_b, d3_b, d4_b, d5_b]

if __name__ == "__main__":
    # Test the network
    model = BDCN(pretrain=True, rate=4).cuda()
    x = torch.randn(1, 3, 500, 500).cuda()
    outputs = model(x)
    print(f"Final fuse output shape: {outputs[0].shape}")
    for i, out in enumerate(outputs[1:], 1):
        print(f"Side output {i} shape: {out.shape}")