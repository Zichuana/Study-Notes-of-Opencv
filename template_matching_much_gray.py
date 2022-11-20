import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('b.jpg', 0)
img = cv2.resize(img, (300, 400))
template = img[70:140, 80:240]
img1 = np.hstack([img, img])
img2 = np.hstack([img, img])
img = np.vstack([img1, img2])
w, h = template.shape[::-1]
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.95  # 调整
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 1)
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()