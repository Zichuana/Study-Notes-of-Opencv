import cv2 as cv
import numpy as np

src = cv.imread("C:/Users/Zichuana/Pictures/Saved Pictures/hhh.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)  # 用于创建显示图像的窗口
cv.imshow("input", src)

# sharpen_op = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]], dtype=np.float32)
sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpen_image = cv.filter2D(src, cv.CV_32F, sharpen_op)
# 这个函数一般是用于图像的卷积，但 OpenCV 文档里说这个函数不完全等于图像卷积,日常用的图像卷积定义和这个函数的功能实际上是一致的。
# 必须指定 参1 src 原图像 参2 ddepth 图像目标深度 参3 kernel 卷积核
# 非必须 anchor（tuple）卷积锚点 delta 偏移量，卷积结果要加上这个数字 borderType 边缘类型
sharpen_image = cv.convertScaleAbs(sharpen_image)  # 该操作可实现图像增强等相关先行操作的快速运算
cv.imshow("sharpen_image", sharpen_image)

h, w = src.shape[:2]
result = np.zeros([h, w*2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w:2*w, :] = sharpen_image
cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "sharpen image", (w+10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("sharpen_image", result)
# cv.imwrite("D:/result.png", result)

cv.waitKey(10000)
cv.destroyAllWindows()
