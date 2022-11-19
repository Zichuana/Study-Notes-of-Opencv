import numpy as np
import imutils
import cv2


# python 3.7
# Package               Version
# --------------------- ---------
# certifi               2022.9.24
# cycler                0.11.0
# fonttools             4.38.0
# imutils               0.5.4
# kiwisolver            1.4.4
# matplotlib            3.5.3
# numpy                 1.21.6
# opencv-contrib-python 3.4.2.16
# opencv-python         3.4.2.16
# packaging             21.3
# Pillow                9.3.0
# pip                   22.2.2
# pyparsing             3.0.9
# python-dateutil       2.8.2
# setuptools            65.5.0
# six                   1.16.0
# typing_extensions     4.4.0
# wheel                 0.37.1
# wincertstore          0.2
class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):

        # 简单地检测关键点并从两个图像中提取局部不变量描述符SIFT并匹配
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 没有足够的匹配关键点，返回空
        if M is None:
            return None

        # 应用透视变换将图像缝合在一起
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检查是否应该可视化关键点匹配
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        # 返回缝合图像
        return result

    def detectAndDescribe(self, image):
        """
        :param image:
        :return: 特征描述点
        """
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 计算两组点之间的单应性需要  至少初始的四组匹配。
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        return None

    # 可视化
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 仅当关键点成功匹配时才处理匹配
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis


if __name__ == '__main__':
    imageA = cv2.imread('b.jpg')
    imageB = cv2.imread('a.jpg')
    print(imageA)
    print(imageB)
    # imageA = cv2.resize(imageA, (400, 300))
    # imageB = cv2.resize(imageB, (400, 300))
    imageA = imutils.resize(imageA, width=400)
    imageB = imutils.resize(imageB, width=400)
    # 将图像缝合在一起以创建全景图
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
