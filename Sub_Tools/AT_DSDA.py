import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        g = F.silu(self.WG(x))
        z = self.W1(x)
        return self.W2(g * z)


class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads, lambda_init):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head))

        self.lambda_init = lambda_init

        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X):
        batch, N, d_model = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)

        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init

        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        mask = torch.tril(torch.ones((N, N), device=X.device)).unsqueeze(0).unsqueeze(0)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling

        A1 = A1 + mask
        A2 = A2 + mask

        attention1 = F.softmax(A1, dim=-1)
        attention2 = F.softmax(A2, dim=-1)
        attention = attention1 - lambda_val * attention2

        O = torch.matmul(attention, V)

        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        O_normalized = O_normalized * (1 - self.lambda_init)

        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)
        out = self.W_o(O_concat)

        return out


class DiffTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, lambda_init):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x):
        y = self.attn(self.norm1(x)) + x
        z = self.ff(self.norm2(y)) + y
        return z


class DiffAttnContextModule(nn.Module):
    def __init__(self, in_channels, num_heads=8, lambda_init=0.8):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.lambda_init = lambda_init
        self.transformer_layer = DiffTransformerLayer(d_model=in_channels, num_heads=num_heads, lambda_init=lambda_init)

    def forward(self, x):
        # 输入 x 的形状: [batch, in_channels, height, width]
        batch, channels, height, width = x.shape

        # Reshape 为 [batch, seq_len, d_model]，其中 seq_len = height * width, d_model = in_channels
        x = x.view(batch, channels, height * width).transpose(1, 2)  # [batch, height * width, in_channels]

        # 通过 DiffTransformerLayer 处理
        x = self.transformer_layer(x)  # [batch, height * width, in_channels]

        # Reshape 回原始形状 [batch, in_channels, height, width]
        x = x.transpose(1, 2).view(batch, channels, height, width)

        return x


if __name__ == '__main__':
    # 测试模块
    in_channels = 64
    num_heads = 4
    lambda_init = 0.8
    mscm = DiffAttnContextModule(in_channels=in_channels, num_heads=num_heads, lambda_init=lambda_init)

    # 模拟图像输入
    input_img = torch.randn(1, in_channels, 64, 64)
    output_img = mscm(input_img)

    print("Input size:", input_img.size())  # [1, 512, 32, 32]
    print("Output size:", output_img.size())  # [1, 512, 32, 32]