import numpy as np
from math import floor, cos, sin, pi, exp
from PIL import Image
import matplotlib.pyplot as plt
from pprint import pprint

# load the source image
srcImg = Image.open("lena.jpg")
srcImg = srcImg.resize((480, 480))
srcImg = np.asarray(srcImg)
print(srcImg.shape)

plt.figure("source img")
plt.imshow(srcImg)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('source img')     # 图像题目
plt.show()


def interpolate(image, subpixel, method="bilinear"):
    if method == "bilinear":
        pixel = [floor(p) for p in subpixel]
        delta = [sub - p for sub, p in zip(subpixel, pixel)]
        surroundings = np.array([image[pixel[0], pixel[1]],
                                image[pixel[0] + 1, pixel[1]],
                                image[pixel[0], pixel[1] + 1],
                                image[pixel[0] + 1, pixel[1] + 1]])
        weight = np.array([(1 - delta[0]) * (1 - delta[1]),
                           delta[0] * (1 - delta[1]),
                           (1 - delta[0]) * delta[1],
                           delta[0] * delta[1]]).reshape([4, 1]).repeat(3, axis=1)
        inter = np.sum(weight * surroundings, axis=0)
        return [int(i) for i in inter]


def transform(image, T):
    image_pad = np.pad(image, ((0, 1), (0, 1), (0, 0)), 'edge')
    src_shape = image.shape
    T_inv = np.linalg.inv(T)
    # calculate image range
    canvas = np.array([[0, src_shape[1], 1],
                       [src_shape[0], 0, 1],
                       [src_shape[0], src_shape[1], 1],
                       [0, 0, 1]])
    canvas = np.transpose(canvas)
    T_canvas = np.trunc(np.matmul(T, canvas))
    T_canvas[0, :] = np.true_divide(T_canvas[0, :], T_canvas[2, :])
    T_canvas[1, :] = np.true_divide(T_canvas[1, :], T_canvas[2, :])

    mins = np.min(T_canvas, axis=1)
    maxs = np.max(T_canvas, axis=1)
    dst_range = [[int(mins[0]), int(maxs[0])], [int(mins[1]), int(maxs[1])]]
    dst_image = 255 * np.ones([dst_range[0][1] - dst_range[0][0], dst_range[1][1] - dst_range[1][0]] + [3])
    # [3]channels

    for x in range(dst_range[0][0], dst_range[0][1]):
        for y in range(dst_range[1][0], dst_range[1][1]):
            T_xy = np.array([x, y])
            T_xy1 = np.array([x, y, 1])
            xy1 = np.matmul(T_inv, T_xy1)
            xy1[0] = xy1[0] / xy1[2]
            xy1[1] = xy1[1] / xy1[2]
            xy = xy1[:2]
            mat_xy = [T_xy[0] - dst_range[0][0], T_xy[1] - dst_range[1][0]]
            if (0 <= xy[0] < src_shape[0] - 1) and (0 <= xy[1] < src_shape[1] - 1):
                dst_image[mat_xy[0], mat_xy[1], :] = np.array(interpolate(image_pad, xy))
            else:
                dst_image[mat_xy[0], mat_xy[1], :] = np.array([255, 255, 255])
    return dst_image, dst_range


def plot(src, ranges, scale=1, title=""):
    plt.figure(figsize=(scale * 5, scale * 5))
    # input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    plt.imshow(src / 255.0)
    plt.title(title)
    # xticks(ticks,labels):
    # ticks:represents the real location of the labels.
    # lables:default-ticks, otherwise-the same things whose length is the same as the ticks
    plt.xticks(range(0, src.shape[1], 100), labels=range(ranges[1][0], ranges[1][1], 100))
    plt.yticks(range(0, src.shape[0], 100), labels=range(ranges[0][0], ranges[0][1], 100))
    plt.show()


# 1.translation transform
def translation(image, t):
    T = np.array([[1, 0, t[0]],
                  [0, 1, t[1]],
                  [0, 0, 1]])
    return transform(image, T)


dstImg, r = translation(srcImg, [50, 50])
print("\ndstImg shape:", dstImg.shape)
plot(dstImg, r, 1, title="translation transform for [50, 50]")


# 2.rotation transform
def rotation(image, theta): # enter radian
    T_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])
    return transform(srcImg, T_rotation)


dstImg, r = rotation(srcImg, 30*np.pi/180)
print("\ndstImg shape:", dstImg.shape)
plot(dstImg, r, title="rotation transform for 30 degree")


# 3.Euclidean transform
def Euclidean(image, theta, t): # enter radian
    T_Euclidean = np.array([[np.cos(theta), -np.sin(theta), t[0]],
                           [np.sin(theta), np.cos(theta), t[1]],
                           [0, 0, 1]])
    return transform(srcImg, T_Euclidean)


dstImg, r = Euclidean(srcImg, -30*np.pi/180, [40, 60])
print("\ndstImg shape:", dstImg.shape)
plot(dstImg, r, title="Euclidean transform for -30 degree & [40, 60]")


# 4.Similarity transform
def similarity(image, theta, t, scale): # enter radian
    T_similarity = np.array([[scale*np.cos(theta), -scale*sin(theta), t[0]],
                             [scale*np.sin(theta), scale*np.cos(theta), t[1]],
                             [0, 0, 1]])
    return transform(srcImg, T_similarity)


dstImg, r = similarity(srcImg, -30*np.pi/180, [40, 60], 0.7)
print("\ndstImg shape:", dstImg.shape)
plot(dstImg, r, title="similarity transform for -30 degree & [40, 60] & scale=0.7")


# 5.Affine transform
def affine(image, A, t): # enter radian
    T_affine = np.array([[A[0][0], A[0][1], t[0]],
                             [A[1][0], A[1][1], t[1]],
                             [0, 0, 1]])
    return transform(srcImg, T_affine)


dstImg, r = affine(srcImg, A=[[1.5, 0.1], [0.2, 1.2]], t=[20, -20])
print("\ndstImg shape:", dstImg.shape)
plot(dstImg, r, title="affine transform for A=[[1.5, 0.1], [0.2, 1.2]], t=[20, -20]")


# 6.projection transform
def projection(image, A, v): # enter radian
    T_projection = np.array([[A[0][0], A[0][1], v[0]],
                             [A[1][0], A[1][1], v[1]],
                             [v[0], v[1], 1]])
    return transform(srcImg, T_projection)


dstImg, r = projection(srcImg, A=[[1.5, 0.1], [0.2, 1.2]], v=[0.001, 0.001])
print("\ndstImg shape:", dstImg.shape)
plot(dstImg, r, title="projection transform for same A, v=[0.001, 0.001]")



