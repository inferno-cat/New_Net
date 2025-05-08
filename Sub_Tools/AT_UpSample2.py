import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample_UP(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super(DySample_UP,self).__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("lp",i.shape)
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("pl",i.shape)
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class DySampleFusion(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4, dyscope=False):
        super(DySampleFusion, self).__init__()
        self.lp_upsampler = DySample_UP(in_channels, scale=scale, style='lp', groups=groups, dyscope=dyscope)
        self.pl_upsampler = DySample_UP(in_channels, scale=scale, style='pl', groups=groups, dyscope=dyscope)
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        lp_output = self.lp_upsampler(x)
        pl_output = self.pl_upsampler(x)
        combined = torch.cat([lp_output, pl_output], dim=1)
        fused_output = self.fusion(combined)
        return fused_output

class DySampleFusion_WithOut(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, groups=4, dyscope=False):
        super(DySampleFusion_WithOut , self).__init__()
        self.lp_upsampler = DySample_UP(in_channels, scale=scale, style='lp', groups=groups, dyscope=dyscope)
        self.pl_upsampler = DySample_UP(in_channels, scale=scale, style='pl', groups=groups, dyscope=dyscope)
        self.fusion = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        lp_output = self.lp_upsampler(x)
        pl_output = self.pl_upsampler(x)
        combined = torch.cat([lp_output, pl_output], dim=1)
        fused_output = self.fusion(combined)
        return fused_output

if __name__ == '__main__':
    input = torch.rand(1, 64, 480, 320)
    # DySample_Fusion = DySampleFusion(in_channels=64, scale=2)
    test = DySampleFusion_WithOut(in_channels=64, out_channels=128, scale=2)
    # output = DySample_Fusion(input)
    output = test(input)
    print('input_size:', input.size())
    print('output_size:', output.size())