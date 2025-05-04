import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# 生成 x 值
x = np.linspace(-5, 5, 1000)

# 创建三张图
plt.figure(figsize=(15, 5))

# ReLU 图
plt.subplot(1, 3, 1)
plt.plot(x, relu(x), 'b-', label='ReLU', linewidth=2)
plt.title('ReLU Function', fontsize=14, pad=10)
plt.xlabel('x', fontsize=12)
plt.ylabel('ReLU(x)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.8)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.8)
plt.legend(fontsize=10)
plt.ylim(-1, 5)

# Sigmoid 图
plt.subplot(1, 3, 2)
plt.plot(x, sigmoid(x), 'r-', label='Sigmoid', linewidth=2)
plt.title('Sigmoid Function', fontsize=14, pad=10)
plt.xlabel('x', fontsize=12)
plt.ylabel('Sigmoid(x)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.8)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.8)
plt.legend(fontsize=10)
plt.ylim(-0.1, 1.1)

# Tanh 图
plt.subplot(1, 3, 3)
plt.plot(x, tanh(x), 'g-', label='Tanh', linewidth=2)
plt.title('Tanh Function', fontsize=14, pad=10)
plt.xlabel('x', fontsize=12)
plt.ylabel('Tanh(x)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.8)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.8)
plt.legend(fontsize=10)
plt.ylim(-1.1, 1.1)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')