# dip lab task 01

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

# 使用 Otsu 算法对原图像进行二值分割
image_path = "./img/图1.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary_image_otsu = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("./img/binary_img_otsu.png", binary_image_otsu)

# 使用移动平均法对原图像进行二值分割
N = 20
c = 0.5


def movingthreshold(f, n, k):
    shape = f.shape
    assert n >= 1
    assert 0 < k < 1
    f[1:-1:2, :] = np.fliplr(f[1:-1:2, :])
    f = f.flatten()
    maf = np.ones(n) / n
    res_filter = lfilter(maf, 1, f)
    g = np.array(f > k * res_filter).astype(int)
    g = g.reshape(shape)
    g[1:-1:2, :] = np.fliplr(g[1:-1:2, :])
    g = g * 255
    return g


image_path = "./img/图1.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
binary_image_moving_avg = movingthreshold(image, N, c)
cv2.imwrite("./img/binary_img_moving_avg.png", binary_image_moving_avg)

# 使用固定阈值对原图像进行二值化分割
image_path = "./img/图1.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary_image_fixed = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
cv2.imwrite("./img/binary_img_fixed.png", binary_image_fixed)

# 显示原始图像和二值化图像
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(binary_image_otsu, cmap="gray")
plt.title("Otsu Binary Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(binary_image_moving_avg, cmap="gray")
plt.title("Moving-Average Binary Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(binary_image_fixed, cmap="gray")
plt.title("Fixed Threshold Binary Image")
plt.axis("off")

plt.show()
