import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

# 局部卷积增强模块
class LocalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LocalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# MLP模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 自注意力模块
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # 注意这里的 x 是 (B, N, C)，要确保这里的维度正确
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 稀疏自注意力模块（SSA）
class SSA(nn.Module):
    def __init__(self, dim, num_heads=4, sparse_size=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.local_conv = LocalConv(dim, dim)  # 局部卷积增强层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sparse_size = sparse_size

    def forward(self, x):
        B, C, H, W = x.shape  # 输入张量形状为 (B, C, H, W)

        # 保存输入用于残差连接
        x_before = x.view(B, C, -1).transpose(1, 2)  # 转换为 (B, N, C)，N = H * W

        # 引入局部卷积增强
        x_local = self.local_conv(x)  # 局部卷积，输出形状仍为 (B, C, H, W)
        x_local = x_local.view(B, C, -1).transpose(1, 2)  # 转换为 (B, N, C)

        # 稀疏自注意力处理
        x = x_before  # 使用转换后的 3D 张量 (B, N, C)
        x = self.norm1(x)  # 应用 LayerNorm，输入为 (B, N, C)
        x = self.attn(x)  # 应用自注意力，输入为 (B, N, C)

        # 合并局部卷积信息
        x = x + x_local  # 融合自注意力输出和局部卷积输出，均为 (B, N, C)
        x = x_before + self.drop_path(x)  # 残差连接

        # MLP 处理
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 应用 MLP

        # 转换回 4D 张量
        x = x.transpose(1, 2).reshape(B, C, H, W)  # 转换为 (B, C, H, W)
        return x

class SAABlocklist(nn.Module):
    def __init__(self, size, dim, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        size_list = [8, 4, 2, 1]
        # size只能是1到4
        assert size in [1, 2, 3, 4], "size must be in [1, 2, 3, 4]"
        # 如果size是4就是sparse_size8 4 2 1 顺次连接
        # 如果size是3就是sparse_size4 2 1
        self.blocks = nn.ModuleList()
        for i in range(size):
            self.blocks.append(SSA(dim, num_heads, sparse_size=size_list[i+(4 - size)],
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                   attn_drop=attn_drop, drop_path=drop_path,
                                   act_layer=act_layer, norm_layer=norm_layer))
        self.conv1x1 = nn.Conv2d(dim * size, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            o.append(x)
        o = torch.cat(o, dim=1)
        o = self.conv1x1(o)
        return o



# 输入 B C H W, 输出 B C H W
if __name__ == "__main__":
    # module = SSA(dim=64)
    module = SAABlocklist(size=4, dim=16)
    input_tensor = torch.randn(2, 16, 60, 40)  # 输入大小为 (B, C, H, W)
    output_tensor = module(input_tensor)
    print('Input size:', input_tensor.size())  # 打印输入张量的形状
    print('Output size:', output_tensor.size())  # 打印输出张量的形状
