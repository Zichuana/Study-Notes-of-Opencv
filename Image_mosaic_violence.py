import cv2

img1 = cv2.imread('b.jpg')
img2 = cv2.imread('a.jpg')
print(img1.shape, img2.shape)
sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.resize(img1, (400, 300))
img2 = cv2.resize(img2, (400, 300))
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 1v1 mapping
bf = cv2.BFMatcher(crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
cv2.imshow('match', img3)
cv2.waitKey(10000)

# multi mapping
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
# cv2.imshow('match k', img4)
# cv2.waitKey()
cv2.imwrite("violence.png", img4)
