import os
import torch
import sys

# 在导入PyTorch之前设置CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 检查可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")

# 如果没有GPU可用，直接退出
if device.type == "cpu":
    print("Error: No GPU available. Exiting.")
    sys.exit(1)

# 检查当前使用的GPU索引
current_gpu = torch.cuda.current_device()
print(f"Current GPU index: {current_gpu}")

# 验证是否为指定的GPU 2
if current_gpu != 0:  # CUDA_VISIBLE_DEVICES=2会将GPU 2映射为索引0
    print(f"Error: Expected GPU 2, but got GPU {current_gpu}. Exiting.")
    sys.exit(1)

# 获取当前GPU的名称以进一步确认
gpu_name = torch.cuda.get_device_name(current_gpu)
print(f"GPU {current_gpu} is {gpu_name}")

# 如果通过检查，打印成功信息
print("Successfully using GPU 2!")