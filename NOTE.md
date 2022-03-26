### 图片输入输出

```python
import cv2
img1 = cv2.imread('C:/Users/zichuana/Desktop/1.jpg')  # 路径
cv2.imshow('t', img1)  # 图片输出
cv2.waitKey(1000)  # 0表示任意键终止，cv2.waitKey(10000)为毫秒级，10000为10秒
cv2.destroyAllWindows()  # 关闭窗口，（）里不指定任何参数，则删除所有窗口，删除特定的窗口，往（）输入特定的窗口值。

img2 = cv2.imread('C:/Users/zichuana/Desktop/1.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图
cv2.imshow('t2', img2)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 保存 cv2.imwrite('',img) 路径，图像
# type(img) 格式
# img.size 像素点个数
# img.dtype 类型

```

### 视频输入输出

```python
vc = cv2.VideoCapture('C:/Users/zichuana/Desktop/1.mp4')
# 检查是否打开正确
if vc.isOpened():
    open, frame = vc.read()  # vc.read() 取帧
else:
    open = False

while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret is True:  # 读帧没有问题
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度
        cv2.imshow('re', gray)  # 展示
        if cv2.waitKey(100) & 0xFF == 27:  # 64位与27比较
            break
# vc.release() ？？？
cv2.destroyAllWindows()
```

### 读取部分图像数据

```python
img = cv2.imread('C:/Users/zichuana/Desktop/1.jpg')
jiequ = img[0:50, 0:200]
cv2.imshow('jiequ', jiequ)
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

### 颜色通道提取

```python
b, g, r = cv2.split(img1)  # b.shape
img1 = cv2.merge((b, g, r))
# 只保留R
cur_img1 = img1.copy()  # ':'通配符
# cur_img1[:, :, :] = 255 白
# cur_img1[:, :, :] = 0 黑
cur_img1[:, :, 0] = 0
cur_img1[:, :, 1] = 0
cv2.imshow('R', cur_img1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# B-0 G-1 R-2
```

### 边界填充

```python
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

replicate = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
constant = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0)
# cv2.imshow('t', replicate)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
plt.subplot(212), plt.imshow(replicate), plt.title('ORIGINAL')  # subplot()内参数表位置
plt.subplot(222), plt.imshow(constant), plt.title('ORIGINAL')
plt.show()
# borderType参数
# BORDER_REPLICATE 复制 复制最边缘像素
# BORDER_REFLECT 反射 hgfedcba|abcdefgh|hgfedcba
# BORDER_REFLECT_101 反射 hgfedcb|abcdefgh|gfedcba
# BORDER_WRAP 外包装 abcdefgh|abcdefgh|abcdefgh
# BORDER_CONSTANT 填充
```

### 图像数值计算

```python
img12 = img1 + 10
print(img1[:5, :, 0])
print(img12[:5, :, 0])
print((img1 + img12)[:5, :, 0])  # 相当于%256
print(cv2.add(img1, img12)[:5, :, 0])  # >=255 -> 255 ; <255 -> x
'''
cv2.imshow('test', img1+img12)
cv2.waitKey(1000)
cv2.destroyAllWindows()

cv2.imshow('test', cv2.add(img1, img12))
cv2.waitKey(1000)
cv2.destroyAllWindows()
'''
```

### 图像融合

```python
img3 = cv2.imread('C:/Users/zichuana/Desktop/2.jpg')
print(img1.shape)
print(img3.shape)
img3 = cv2.resize(img3, (835, 501))  # img3 = cv2.resize(img3, (0, 0), fx=?, fy=?) 比例放缩
print(img3.shape)
res = cv2.addWeighted(img1, 0.4, img3, 0.6, 0)
cv2.imshow('test', res)
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

### 图像阈值

```python
'''
ret, dst = cv2.threshold(src, thresh, maxval, type)
src:输入图，只能输入单通道图像，通常来说是灰度图
dst:输出图
thresh:阈值
maxval:当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
type:二值化操作的类型，包括以下5种类型
1. cv2.THRESH_BINARY 超过阈值部分取最大值，否则取0
2. cv2.THRESH_BINARY_INV 1的反转
3. cv2.THRESH_TRUNC 大于阈值部分设为阈值，否则不变
4. cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为0
5. cv2.THRESH_TOZERO_INV 4的反转
'''
ret, thresh1 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img2, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img2, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img2, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['1', '2', '3', '4', '5']
images = [thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(5):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])  # 警用刻度以及标签
plt.show()
```

### 图像平滑处理

```python
import cv2
import matplotlib.pyplot as plt

# 图像平滑处理（通俗就是给图像去除’噪音‘）
img1 = cv2.imread('C:/Users/zichuana/Desktop/3.jpg')  # 原始图像输入输出
cv2.imshow('img1', img1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 均值滤波，简单的平均卷积操作
bl1 = cv2.blur(img1, (3, 3))  # 矩阵参数最好为奇数
cv2.imshow('bl1', bl1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 方框滤波
# 该情况下和均值滤波一样
box1 = cv2.boxFilter(img1, -1, (3, 3), normalize=True)  # 参数-1表示颜色通道一致，通常情况下是-1
cv2.imshow('box1', box1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# normalize=False情况
box2 = cv2.boxFilter(img1, -1, (3, 3), normalize=False)  # >=255 白色
cv2.imshow('box2', box2)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 高斯与中值滤波
# 高斯（根据权重矩阵，距离越近的作用越大, 同样也是去除噪音点）
Gauss = cv2.GaussianBlur(img1, (5, 5), 1)
cv2.imshow('Gauss', Gauss)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 中值滤波 （去排序取中位数为中心）除滤波操作的话 这个是最好的
median = cv2.medianBlur(img1, 5)
cv2.imshow("median", median)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 一次性拼接展示所有图像
res = np.hstack((bl1, box1, Gauss, median))
cv2.imshow("res", res)
cv2.waitKey(0)
cv2.destroyAllWindows()


```

### 型态学 腐蚀操作

```python
# 型态学 腐蚀操作 （假设是一个手臂的近照片，上面有毛毛，可以除去毛毛）
img2 = cv2.imread('C:/Users/zichuana/Desktop/4.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img2', img2)
cv2.waitKey(1000)
cv2.destroyAllWindows()

kn1 = np.ones((5, 5), np.uint8)
es1 = cv2.erode(img2, kn1, iterations=1)  # iterations 迭代次数
cv2.imshow("es1", es1)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 膨胀操作，腐蚀操作逆运算，可以弥补腐蚀操作带来的损害
# 0 -> 255 腐蚀，255 -> 0 膨胀
kn1 = np.ones((5, 5), np.uint8)
dl1 = cv2.dilate(es1, kn1, iterations=1)  # iterations 迭代次数
cv2.imshow("es1", es1)
cv2.waitKey(1000)
cv2.destroyAllWindows()

kn1 = np.ones((5, 5), np.uint8)
es1 = cv2.erode(img2, kn1, iterations=3)  # iterations 迭代次数
cv2.imshow("es1", es1)
cv2.waitKey(1000)
cv2.destroyAllWindows()

kn1 = np.ones((5, 5), np.uint8)
dl1 = cv2.dilate(es1, kn1, iterations=3)  # iterations 迭代次数
cv2.imshow("es1", es1)
cv2.waitKey(1000)
cv2.destroyAllWindows()

ret, img3 = cv2.threshold(img2, 177, 255, cv2.THRESH_BINARY)
cv2.imshow('img3', img3)
cv2.waitKey(1000)
cv2.destroyAllWindows()

kn2 = np.ones((5, 5), np.uint8)
es2 = cv2.erode(img3, kn2, iterations=1)  # iterations 迭代次数
cv2.imshow("es2", es2)
cv2.waitKey(1000)
cv2.destroyAllWindows()

img4 = cv2.imread('C:/Users/zichuana/Desktop/4.jpg')
cv2.imshow("img4", img4)
cv2.waitKey(1000)
cv2.destroyAllWindows()

kn3 = np.ones((5, 5), np.uint8)
es3 = cv2.erode(img3, kn3, iterations=1)  # iterations 迭代次数 色彩腐蚀次数
# 类似与水的浸润 越迭代次数越多，越浸蚀
cv2.imshow("es2", es3)
cv2.waitKey(1000)
cv2.destroyAllWindows()  # 非灰度图 出现的图片接近0

```

### 开闭运算

```python
iimport cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("C:/Users/zichuana/Desktop/1.jpg", cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh2 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh1", thresh1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
cv2.imshow("thresh2", thresh2)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 开运算 先腐蚀再膨胀 255 -> 0 -> 255
mid1 = np.ones((5, 5), np.uint8)  # unit8 图片矩阵适使用的数据类型
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, mid1)
cv2.imshow("opening", opening)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 闭运算 先膨胀再腐蚀
closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, mid1)
cv2.imshow("closing", closing)
cv2.waitKey(1000)
cv2.destroyAllWindows()

opening2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, mid1)
cv2.imshow("opening", opening2)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 闭运算 先膨胀再腐蚀 0 -> 255 -> 0
closing2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, mid1)
cv2.imshow("closing", closing2)
cv2.waitKey(1000)
cv2.destroyAllWindows()

res1 = np.hstack((thresh1, opening, closing))
cv2.imshow("res1", res1)
cv2.waitKey(1000)
cv2.destroyAllWindows()
res2 = np.hstack((thresh2, opening2, closing2))
cv2.imshow("res2", res2)
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

### 梯度计算

```python

# 梯度计算
mid2 = np.ones((7, 7), np.uint8)
result1 = cv2.dilate(thresh2, mid2, iterations=3)
result2 = cv2.erode(thresh2, mid2, iterations=3)
res3 = np.hstack((result1, result2))
cv2.imshow("res3", res3)
cv2.waitKey(1000)
cv2.destroyAllWindows()

gra = cv2.morphologyEx(thresh2, cv2.MORPH_GRADIENT, mid2)
cv2.imshow("gra", gra)
cv2.waitKey(1000)
cv2.destroyAllWindows()
```

