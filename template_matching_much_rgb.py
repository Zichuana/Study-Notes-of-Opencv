import cv2
import numpy
import numpy as np

# 读取目标图片
target = cv2.imread("b.jpg")
target = cv2.resize(target, (300, 400))
img1 = np.hstack([target, target])
img2 = np.hstack([target, target])
target = np.vstack([img1, img2])
# 读取模板图片
template = target[70:140, 80:240]
cv2.imshow('template', template)
cv2.waitKey(1000)
# 获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]
# 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
# 归一化处理
# cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
# 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# 绘制矩形边框，将匹配区域标注出来
# min_loc：矩形定点
# (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
# (0,0,225)：矩形的边框颜色；2：矩形边框宽度
cv2.rectangle(target, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 225, 0), 2)
# 匹配值转换为字符串
# 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
# 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)
# 初始化位置参数
temp_loc = min_loc
other_loc = min_loc
numOfloc = 1
# 第一次筛选----规定匹配阈值，将满足阈值的从result中提取出来
# 源博客“对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法设置匹配阈值为0.01”！！！需要调整参数①
threshold = 0.0001
loc = numpy.where(result < threshold)
print(loc)
# 遍历提取出来的位置
for other_loc in zip(*loc[::-1]):
    # 第二次筛选----源博客：“将位置偏移小于5个像素的结果舍去”！！！需要调整的参数②。
    if (temp_loc[0] + 5 < other_loc[0]) or (temp_loc[1] + 5 < other_loc[1]):
        numOfloc = numOfloc + 1
        temp_loc = other_loc
        cv2.rectangle(target, other_loc, (other_loc[0] + twidth, other_loc[1] + theight), (0, 0, 225), 2)
    # numOfloc = numOfloc + 1
    # temp_loc = other_loc
    # cv2.rectangle(target, other_loc, (other_loc[0] + twidth, other_loc[1] + theight), (0, 0, 225), 2)

str_numOfloc = str(numOfloc)
# 显示结果,并将匹配值显示在标题栏上
strText = "MatchResult----MatchingValue=" + strmin_val + "----NumberOfPosition=" + str_numOfloc
cv2.imshow(strText, target)
cv2.waitKey(10000)
