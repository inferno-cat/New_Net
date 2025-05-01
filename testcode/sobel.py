import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r'D:\rawcode\data\BSDS500_flipped_rotated_pad\images\test\108069.jpg', cv2.IMREAD_GRAYSCALE)


# 检查图像是否成功加载
if image is None:
    raise Exception("Error: Could not load the image.")

# 高斯模糊以减少噪声
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Sobel 边缘检测
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # 合并梯度

# 归一化以显示
sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 显示结果
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Detection')
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.tight_layout()
# 保存结果图像
plt.savefig('sobel_edge_detection_result.png')