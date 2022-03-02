### 图片输入输出

```python
import cv2
img = cv2.imread('C:/Users/zichuana/Desktop/1.jpg')  # 路径
cv2.imshow('t', img)  # 图片输出
cv2.waitKey(1000)  # 0表示任意键终止，cv2.waitKey(10000)为毫秒级，10000为10秒
cv2.destroyAllWindows()  # 关闭窗口，（）里不指定任何参数，则删除所有窗口，删除特定的窗口，往（）输入特定的窗口值。

img = cv2.imread('C:/Users/zichuana/Desktop/1.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图
cv2.imshow('t2', img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 保存 cv2.imwrite('',img) 路径，图像
# type(img) 格式
# img.size 像素点个数
# img.dtype 类型

```

