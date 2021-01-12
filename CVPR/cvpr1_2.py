import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

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

Gaussia_kernel = (1/256) * np.array([[1, 4, 6, 4, 1],
                                     [4, 16, 24, 16, 4],
                                     [6, 24, 36, 24, 6],
                                     [4, 16, 24, 16, 4],
                                     [1, 4, 6, 4, 1]])


def conv(image, kernel):
    kernel_size = kernel.shape[0]
    image_pad = np.zeros([image.shape[0]+4, image.shape[1]+4] + [3])
    kernels = np.zeros([kernel_size, kernel_size] + [3])
    for i in range(3):
        image_pad[:, :, i] = np.pad(image[:, :, i], ((2, 2), (2, 2)), 'edge')
        kernels[:, :, i] = kernel

    dst_image = np.zeros([image.shape[0], image.shape[1]] + [3])
    dst_shape = dst_image.shape
    for x in range(dst_shape[0]):
        for y in range(dst_shape[1]):
            surroudings = image_pad[x:x + kernel_size, y:y + kernel_size, :]
            conv_rslt = surroudings * kernels
            dst_image[x, y, :] = np.sum(np.sum(conv_rslt, axis=0), axis=0)
    return dst_image


def pyrDown(image, kernel, step=2):
    image_conv = conv(image, kernel)
    image_down = image_conv[::step, ::step, :]  # interval is step
    return image_down


def pyrUp(image, kernel, step=2):
    src_shape = image.shape
    image_up = np.zeros([src_shape[0]*2, src_shape[1]*2] + [3])
    for x in range(src_shape[0]):
        for y in range(src_shape[1]):
            image_up[2*x+1, 2*y+1, :] = image[x, y, :]
    image_conv = conv(image_up, kernel)
    image_up[::2, ::2, :] = image_conv[::step, ::step, :]
    return image_conv

# build Gaussian pyramid
GauPry = [srcImg]
num_layer = 5
img = srcImg
for _ in range(num_layer):
    image_down = pyrDown(img, Gaussia_kernel)
    GauPry.append(image_down)
    img = image_down

plt.figure(figsize=(10, 7), dpi=48)
# figure, ax = plt.subplots(2, 3, sharex=True, sharey=True)
for i in range(len(GauPry)):
    plt.subplot(2, 3, i+1)
    plt.imshow(GauPry[i]/255.0)
    plt.title('G' + str(i))
plt.show()

# changing pyrDown step
GauPry_step = [srcImg]
steps = [2, 3, 5, 10, 20]
for step in steps:
    image_down = pyrDown(srcImg, Gaussia_kernel, step)
    GauPry_step.append(image_down)

plt.figure(figsize=(25, 25))
plt.subplot(2, 3, 1)
plt.imshow(GauPry_step[0]/255.)
plt.title('Original image')
for i in range(len(steps)):
    plt.subplot(2, 3, i+2)
    plt.imshow(GauPry_step[i+1]/255.)
    plt.title('step ' + str(steps[i]))
plt.show()

# Laplace pyramid
srcPry = [srcImg]
num_layer = 5
src_image = srcImg
for _ in range(num_layer):
    src_down = pyrDown(src_image, Gaussia_kernel)
    srcPry.append(src_down)
    src_image = src_down

srcExpand = []
for i in range(num_layer):
    # Gaussian核*4是因为在扩充了2*2，相比原来的元素就要少4倍
    src_up = pyrUp(srcPry[i + 1], 2 * 2 * Gaussia_kernel)
    srcExpand.append(src_up)
LapPry = []

for i in range(num_layer):
    LapPry.append(srcPry[i] - srcExpand[i])
LapPry.append(srcPry[-1])

plt.figure(figsize=(10, 7))
for i in range(len(LapPry)):
    plt.subplot(2, 3, i + 1)
    if i == 5:
        plt.imshow(LapPry[i] / 255.)
    else:
        plt.imshow(LapPry[i])
    plt.title('G ' + str(i))
plt.show()
