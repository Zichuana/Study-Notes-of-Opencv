import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('b.jpg', 0)
img = cv2.resize(img, (300, 400))
# template = cv2.imread('c.png', 0)
template = img[70:140, 80:240]
print(img.shape)
print(template.shape)
th, tw = template.shape[::]
# tw, th = template.shape[::-1]
rv = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
print(rv)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
topLeft = minLoc
bottomRight = (topLeft[0] + tw, topLeft[1] + th)
img = cv2.rectangle(img, topLeft, bottomRight, 255, 2)  # 绘制矩形
# img：指定一张图片，在这张图片的基础上进行绘制；（img相当于一个画板）
# pt1： 由（x_min，x_max）组成，为绘制的边框的左上角；
# pt2： 由（x_max, y_max）坐标，为绘制的边框的右下角，示意如下：
# color：指定边框的颜色，由（B,G,R）组成，当为（255,0，0）时为绿色，可以自由设定；
# thinkness：线条的粗细值，为正值时代表线条的粗细（以像素为单位），为负值时边框实心;
plt.subplot(131), plt.imshow(template, cmap='gray')
plt.title('template'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(rv, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()
