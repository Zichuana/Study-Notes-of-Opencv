import cv2
from matplotlib import pyplot as plt

# 读取目标图片
target = cv2.imread("b.jpg")
target = cv2.resize(target, (300, 400))
# 读取模板图片
template = target[70:140, 80:240]
# 获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]
# 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
# 归一化处理
cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
# 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# 匹配值转换为字符串
# 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
# 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)
# 绘制矩形边框，将匹配区域标注出来
# min_loc：矩形定点
# (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
# (0,0,225)：矩形的边框颜色；2：矩形边框宽度
cv2.rectangle(target, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)
# 显示结果,并将匹配值显示在标题栏上
cv2.imshow("MatchResult----MatchingValue=" + strmin_val, target)
cv2.waitKey(1000)
# cv2.destroyAllWindows()
plt.subplot(131), plt.imshow(template)
plt.title('template'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(result)
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(target)
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()
# img = Image.fromarray(np.uint8(img))
# print(img.shape)
b, g, r = cv2.split(target)
img = cv2.merge([r, g, b])
plt.imshow(img)
plt.axis('off')  # 关掉坐标轴为 off
plt.title('target')  # 图像题目
plt.show()
