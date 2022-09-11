import cv2
import numpy as np

# 一、基础读取与显示
# 1. 读取与显示图片
img = cv2.imread("C:/Users/Zichuana/Pictures/Saved Pictures/hhh.jpg")
cv2.imshow("img", img)
cv2.waitKey(100)
cv2.destroyAllWindows()

print(img.shape)

# 2. 读取与显示视频
vc = cv2.VideoCapture("C:/Users/Zichuana/Desktop/QQ短视频20220911130125.mp4")
if vc.isOpened():
    open, frame = vc.read()  # vc.read() 取帧
else:
    open = False

while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret is True:  # 读帧没有问题
        cv2.imshow('re', frame)  # 展示
        if cv2.waitKey(10) & 0xFF == 27:  # 64位与27比较
            break
# 释放捕获器
vc.release()
cv2.destroyAllWindows()

# 二、基本图像操作
# 1. 缩放
# 简单缩放
dst = cv2.resize(img, (400, 500))
cv2.imshow("test: %d x %d" % (dst.shape[0], dst.shape[1]), dst)
cv2.waitKey(10)
cv2.destroyAllWindows()
# 比例缩放
dst = cv2.resize(img, (0, 0), None, 0.5, 0.5)  # dst = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
# 优先级： dsize > fx fy 当(0, 0)不合法时 执行fx, fy
cv2.imshow("test: %d x %d" % (dst.shape[0], dst.shape[1]), dst)
cv2.waitKey(10)
cv2.destroyAllWindows()

# 2. 裁剪
cropped_image = img[100:900, 100:900]
cv2.imshow("crop %d x %d" % (cropped_image.shape[0], cropped_image.shape[1]), cropped_image)
print(cropped_image.shape)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# 3. 画点
point = (500, 500)
clone_img = img.copy()
cv2.circle(clone_img, point, 1, (255, 0, 0), 2)
# cv2.circle(img, 点坐标, 点大小, 颜色, 边框线条大小)
print(clone_img.shape)
cv2.imshow("point", clone_img)
cv2.waitKey(100)
cv2.destroyAllWindows()
# 4. 画圆 cv2.circle(img, 圆心坐标, 半径, 颜色, 边框线条大小)
clone_img = img.copy()
cv2.circle(clone_img, (500, 500), 100, (0, 255, 0), 4)
cv2.imshow("yuanquan", clone_img)
cv2.waitKey(100)
cv2.destroyAllWindows()
# 5. 矩形
clone_img = img.copy()
cv2.rectangle(clone_img, (200, 200), (300, 300), color=(255, 0, 255), thickness=10)
cv2.imshow("sanjiaoxing", clone_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# 画线 cv2.line(img, (200, 200), (200, 300), color=(255, 0, 255), thickness=2)


# 三、基本图像滤波
# 1. 高斯滤波
def add_gauss_noise(image, mean=0, val=0.0):  # 对图像添加高斯噪声
    size = image.shape
    image = image / 255
    gauss = np.random.normal(mean, val**0.05, size)
    image = image + gauss
    return image


gauss_img = add_gauss_noise(img)
cv2.imshow("gauss_img", gauss_img)
cv2.waitKey(10)
cv2.destroyAllWindows()

Gaussian_image = cv2.GaussianBlur(gauss_img, (5, 5), 1, 2)  # (3,3)高斯核，必须为正的奇数，可以不同
# sigmaX 是卷积核在水平方向上（X 轴方向）的标准差，其控制的是权重比例。 sigmaY是卷积核在垂直方向上（Y轴方向）的标准差。如果将该值设置为0，则只采用sigmaX的值
cv2.imshow("Gaussian_image", Gaussian_image)
cv2.waitKey(10)
cv2.destroyAllWindows()


# 2. 中值滤波
def add_peppersalt_noise(image, n=10000):  # 生成椒盐噪声
    result = image.copy()
    # 测量图片的长和宽
    w, h =image.shape[:2]
    # 生成n个椒盐噪声
    for i in range(n):
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        if np.random.randint(0, 2) == 0:
            result[x, y] = 0
        else:
            result[x, y] = 255
    return result


pepper_img = add_peppersalt_noise(img)
cv2.imshow("median_img", pepper_img)
cv2.waitKey(10)
cv2.destroyAllWindows()

median = cv2.medianBlur(pepper_img, 5)  # 核为正方形 中值替代
cv2.imshow("median", median)
cv2.waitKey(10)
cv2.destroyAllWindows()


# 3. 平均滤波（均值）
blur_image = cv2.blur(pepper_img, (5, 5))  # 核最好为奇数
# 其他参数
#  anchor 是锚点，其默认值是（-1,-1），表示当前计算均值的点位于核的中心点位置。该值使用默认值即可，在特殊情况下可以指定不同的点作为锚点。
#  borderType是边界样式，该值决定了以何种方式处理边界。一般情况下不需要考虑该值的取值，直接采用默认值即可。
cv2.imshow("blur_image", blur_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()


# 4. 图像锐化
