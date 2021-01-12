import cv2
import numpy as np


ROTATE = 0
AFFINE = 1

img = cv2.imread('corner.png')
rows, cols, channel = img.shape
# Rotate
if ROTATE == 1:
    T = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
    img = cv2.warpAffine(img, T, (cols, rows))

# Affine
if AFFINE == 1:
    M = np.float32([[1, 0.1, 20], [0.1, 1, 10]])
    img = cv2.warpAffine(img, M, (cols, rows))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 4, 3, 0.04)
dst = cv2.dilate(dst, None)
ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)
img[dst > 0.01*dst.max()] = [0, 0, 255]
cv2.imshow('Harris', img)

# 找到质心
ret, labels, states, centroids = cv2.connectedComponentsWithStats(dst)
# 定义停止和改进角落的标注
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# 现在绘制它们
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0, 0, 255]  # 红色
img[res[:, 3], res[:, 2]] = [0, 255, 0]  # 绿色

# cv2.imwrite('subpixel5.png', img)
cv2.imshow('res', img)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
