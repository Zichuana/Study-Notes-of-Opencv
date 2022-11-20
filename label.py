import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# English
img = cv2.imread('a.jpg')
# 添加的文字
text = 'A'
# 文字起始的位置
position = (800, 1000)
# 字体大小
font_scale = 50
# 字体颜色
font_color = (0, 0, 255)
# 默认字体
font = cv2.FONT_HERSHEY_SIMPLEX
# 线的粗细
thickness = 10
cv2.putText(img, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
# chinese
# 判断是否OpenCV图片类型，也就是numpy.ndarray数据类型
if (isinstance(img, np.ndarray)):
    # 把img的numpy.ndarray数据类型格式化为PLI的Image图像数据类型
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)

    print("img type is {}".format(type(img)))  # 此时观察到img数据类型是 PIL.Image.Image

    # 加载字体
    fontText = ImageFont.truetype("simsun.ttc", 66, encoding="utf-8")
    # 在图像上添加字符，中英文皆可
    # 参数概述draw.text(坐标xy，添加文字，文字颜色，字体)
    draw.text((100, 40), '啦啦啦！', (255, 100, 200), font=fontText)
    # 转换回OpenCV可处理的图片类型，这样之后可以继续使用OpenCV或深度学习对图像进行树立
    img = np.asarray(img)

print("img type is {}".format(type(img)))  # 此时观察到img数据类型是 numpy.ndarray
plt.imshow(img)
plt.show()