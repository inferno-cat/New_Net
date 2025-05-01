class LayerCalculator:
    def __init__(self, layer_type, input_shape, kernel_shape, out_channels=None, stride=1, padding=0):
        """
        初始化层参数
        :param layer_type: 层类型 ('conv', 'depthwise_separable', 'maxpool', 'avgpool')
        :param input_shape: 输入形状 (B, C_in, H_in, W_in)
        :param kernel_shape: 卷积核或池化核形状 (K_h, K_w)
        :param out_channels: 输出通道数 C_out（卷积和深度可分离卷积需要，池化层忽略）
        :param stride: 步幅（支持单一值或 (stride_h, stride_w)）
        :param padding: 填充（支持单一值或 (padding_h, padding_w)）
        """
        self.layer_type = layer_type.lower()
        self.B, self.C_in, self.H_in, self.W_in = input_shape
        self.K_h, self.K_w = kernel_shape
        self.C_out = out_channels if layer_type in ['conv', 'depthwise_separable'] else self.C_in
        self.S_h = stride if isinstance(stride, int) else stride[0]
        self.S_w = stride if isinstance(stride, int) else stride[1]
        self.P_h = padding if isinstance(padding, int) else padding[0]
        self.P_w = padding if isinstance(padding, int) else padding[1]

        if self.layer_type not in ['conv', 'depthwise_separable', 'maxpool', 'avgpool']:
            raise ValueError("layer_type must be 'conv', 'depthwise_separable', 'maxpool', or 'avgpool'")
        if self.layer_type in ['conv', 'depthwise_separable'] and out_channels is None:
            raise ValueError("out_channels must be specified for conv or depthwise_separable layer")

    def calculate_output_shape(self):
        """计算输出特征图形状 (B, C_out, H_out, W_out)"""
        H_out = (self.H_in - self.K_h + 2 * self.P_h) // self.S_h + 1
        W_out = (self.W_in - self.K_w + 2 * self.P_w) // self.S_w + 1
        return (self.B, self.C_out, H_out, W_out)

    def calculate_params(self):
        """计算参数量"""
        if self.layer_type == 'conv':
            return (self.K_h * self.K_w * self.C_in * self.C_out) + self.C_out
        elif self.layer_type == 'depthwise_separable':
            # 深度卷积：K_h * K_w * C_in + C_in（偏置）
            # 逐点卷积：1 * 1 * C_in * C_out + C_out（偏置）
            depthwise_params = (self.K_h * self.K_w * self.C_in) + self.C_in
            pointwise_params = (self.C_in * self.C_out) + self.C_out
            return depthwise_params + pointwise_params
        return 0  # 池化层无参数

    def calculate_flops(self):
        """计算推理FLOPs"""
        _, _, H_out, W_out = self.calculate_output_shape()
        if self.layer_type == 'conv':
            conv_flops = 2 * self.B * H_out * W_out * self.C_out * self.C_in * self.K_h * self.K_w
            bias_flops = self.B * H_out * W_out * self.C_out
            return conv_flops + bias_flops
        elif self.layer_type == 'depthwise_separable':
            # 深度卷积：每个通道独立卷积
            depthwise_flops = 2 * self.B * H_out * W_out * self.C_in * self.K_h * self.K_w
            depthwise_bias = self.B * H_out * W_out * self.C_in
            # 逐点卷积：1x1卷积
            pointwise_flops = 2 * self.B * H_out * W_out * self.C_out * self.C_in
            pointwise_bias = self.B * H_out * W_out * self.C_out
            return depthwise_flops + depthwise_bias + pointwise_flops + pointwise_bias
        elif self.layer_type == 'maxpool':
            return self.B * H_out * W_out * self.C_out * (self.K_h * self.K_w - 1)
        elif self.layer_type == 'avgpool':
            return self.B * H_out * W_out * self.C_out * self.K_h * self.K_w

    def calculate_training_flops(self):
        """计算训练FLOPs"""
        if self.layer_type in ['conv', 'depthwise_separable']:
            return 3 * self.calculate_flops()
        return self.calculate_flops()  # 池化层训练FLOPs约等于推理FLOPs

    def format_number(self, number):
        """将大数值转换为带单位的字符串（K, M, G）"""
        if number >= 1e9:
            return f"{number / 1e9:.2f}G"
        elif number >= 1e6:
            return f"{number / 1e6:.2f}M"
        elif number >= 1e3:
            return f"{number / 1e3:.2f}K"
        else:
            return f"{number:.2f}"

    def summary(self):
        """打印层信息，包含原始数值和带单位格式"""
        B, C_out, H_out, W_out = self.calculate_output_shape()
        params = self.calculate_params()
        flops = self.calculate_flops()
        training_flops = self.calculate_training_flops()

        summary_str = (
            f"{self.layer_type.replace('_', ' ').title()} Layer Summary:\n"
            f"Input Shape: (B={self.B}, C={self.C_in}, H={self.H_in}, W={self.W_in})\n"
            f"Output Shape: (B={B}, C={C_out}, H={H_out}, W={W_out})\n"
            f"Kernel Shape: ({self.K_h}, {self.K_w})\n"
            f"Stride: ({self.S_h}, {self.S_w})\n"
            f"Padding: ({self.P_h}, {self.P_w})\n"
            f"Parameters: {params:,} ({self.format_number(params)})\n"
            f"Inference FLOPs: {flops:,} ({self.format_number(flops)})\n"
            f"Training FLOPs: {training_flops:,} ({self.format_number(training_flops)})\n"
        )
        return summary_str


# 示例用法
if __name__ == "__main__":
    # 示例1：标准卷积层 (B=12, C=16, H=480, W=320)，3x3核，输出通道32，步幅1，填充1
    conv_layer = LayerCalculator(
        layer_type='conv',
        input_shape=(12, 16, 480, 320),
        kernel_shape=(3, 3),
        out_channels=32,
        stride=1,
        padding=1
    )
    print(conv_layer.summary())

    # 示例2：深度可分离卷积层 (B=12, C=16, H=480, W=320)，3x3核，输出通道32，步幅1，填充1
    depthwise_separable_layer = LayerCalculator(
        layer_type='depthwise_separable',
        input_shape=(12, 16, 480, 320),
        kernel_shape=(3, 3),
        out_channels=32,
        stride=1,
        padding=1
    )
    print(depthwise_separable_layer.summary())

    # 示例3：最大池化层 (B=12, C=16, H=480, W=320)，2x2核，步幅2，填充0
    maxpool_layer = LayerCalculator(
        layer_type='maxpool',
        input_shape=(12, 16, 480, 320),
        kernel_shape=(2, 2),
        out_channels=None,
        stride=2,
        padding=0
    )
    print(maxpool_layer.summary())

    # 示例4：平均池化层 (B=12, C=16, H=480, W=320)，2x2核，步幅2，填充0
    avgpool_layer = LayerCalculator(
        layer_type='avgpool',
        input_shape=(12, 16, 480, 320),
        kernel_shape=(2, 2),
        out_channels=None,
        stride=2,
        padding=0
    )
    print(avgpool_layer.summary())