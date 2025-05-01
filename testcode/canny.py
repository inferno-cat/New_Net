import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r'D:\rawcode\data\BSDS500_flipped_rotated_pad\images\test\108069.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    raise Exception("Error: Could not load the image.")

# 1. 高斯模糊以减少噪声
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 2. Canny 边缘检测
# 低阈值和高阈值可以根据需要调整
edges = cv2.Canny(blurred, 100, 200)

# 3. 显示结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Canny Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
# 保存结果图像
plt.savefig('edge_detection_result.png')