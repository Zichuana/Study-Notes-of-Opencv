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

