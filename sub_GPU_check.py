# 在导入PyTorch之前设置只使用GPU 4
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import subprocess
import sys
def check_gpu_memory(gpu_id=4, max_usage_ratio=0.1):
    """
    检查指定GPU的显存占用率。
    参数：
        gpu_id: 要检查的GPU编号
        max_usage_ratio: 最大允许的显存占用率（例如0.1表示10%）
    返回：
        True如果GPU空闲（显存占用率低于max_usage_ratio），否则False
    """
    try:
        # 调用nvidia-smi获取GPU显存信息
        result = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.used,memory.total --format=csv',
            shell=True
        ).decode('utf-8')

        # 解析nvidia-smi输出
        lines = result.strip().split('\n')[1:]  # 跳过表头
        if gpu_id >= len(lines):
            print(f"Error: GPU {gpu_id} does not exist.")
            return False

        # 获取指定GPU的显存使用情况
        used, total = map(lambda x: int(x.split()[0]), lines[gpu_id].split(','))
        usage_ratio = used / total
        print(f"GPU {gpu_id} memory usage: {used}MiB/{total}MiB ({usage_ratio:.2%})")

        # 检查显存占用率
        if usage_ratio > max_usage_ratio:
            print(f"Error: GPU {gpu_id} is in use (memory usage > {max_usage_ratio:.0%}).")
            return False
        return True
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return False


# # 检查GPU 4是否空闲
# if not check_gpu_memory(gpu_id=4, max_usage_ratio=0.1):
#     print("Exiting program due to GPU 4 being in use.")
#     sys.exit(1)
#
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Selected device: {device}")