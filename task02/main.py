# dip lab task 02

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_gray = cv2.imread("./img/图2.tif", cv2.IMREAD_GRAYSCALE)
img_blur = cv2.GaussianBlur(
    img_gray, (9, 9), sigmaX=2, borderType=cv2.BORDER_DEFAULT
)

ret, img_bin_noblur = cv2.threshold(
    img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
ret, img_bin_blur = cv2.threshold(
    img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

img_circle = cv2.imread("./img/图2.tif", cv2.IMREAD_COLOR)
circles = cv2.HoughCircles(
    img_bin_blur, cv2.HOUGH_GRADIENT,
    1, 10, param1=50, param2=20, minRadius=2, maxRadius=100
)

diameter_output = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center_x, center_y, radius = i[0], i[1], i[2]
        diameter = 2 * radius
        print(
            f"Center of the circle: ({center_x}, {center_y}),",
            f"diameter: {diameter}"
        )
        diameter_output.append(diameter)
        # 绘制圆和圆心
        if diameter < 50:
            cv2.circle(
                img_circle, (center_x, center_y),
                radius, (255, 0, 255), 3
            )  # 轮廓
            cv2.circle(
                img_circle, (center_x, center_y),
                2, (0, 0, 255), 6
            )  # 圆心
        elif 50 < diameter < 90:
            cv2.circle(
                img_circle, (center_x, center_y),
                radius, (0, 255, 255), 3
            )  # 轮廓
            cv2.circle(
                img_circle, (center_x, center_y),
                2, (0, 0, 255), 6
            )  # 圆心
else:
    print("[WARN] No circle wooden nails found.")

plt.subplot(2, 3, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(img_blur, cmap="gray")
plt.title("Gaussian Blur Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(img_bin_noblur, cmap="gray")
plt.title("Binary Image (No Blur)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(img_bin_blur, cmap="gray")
plt.title("Binary Image (Blur)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_circle, cmap="gray")
plt.title("Detected Wooden Nails")
plt.axis("on")

plt.subplot(2, 3, 6)
plt.hist(diameter_output, color="purple")
plt.title("Histogram of the Nails' Diameter")
plt.axis("on")

plt.show()
