import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

# load the source image
srcImg = Image.open("lena.jpg")
srcImg = srcImg.resize((480, 480))
srcImg = np.asarray(srcImg)
print(srcImg.shape)


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


def gaussian_dx(sigma, x, y):
    gaussian_xy = 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/(2*sigma**2))
    return -x/(sigma**2) * gaussian_xy


def gaussian_dy(sigma, x, y):
    gaussian_xy = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return -y / (sigma**2) * gaussian_xy


def get_gaussian_kernel(sigma, kernel_size, direction):
    Gaussian_d_kernel = np.zeros([kernel_size, kernel_size])
    for x in range(kernel_size):
        for y in range(kernel_size):
            if direction == 'x':
                Gaussian_d_kernel[x, y] = gaussian_dx(sigma, x - kernel_size // 2, y - kernel_size // 2)
            elif direction == 'y':
                Gaussian_d_kernel[x, y] = gaussian_dy(sigma, x - kernel_size // 2, y - kernel_size // 2)
    return Gaussian_d_kernel


def get_gaussian_derivative(image, sigma):
    _dx_kernel = get_gaussian_kernel(sigma, 5, 'x')
    _dy_kernel = get_gaussian_kernel(sigma, 5, 'y')
    _image_dx = conv(image, _dx_kernel)
    _image_dy = conv(image, _dy_kernel)
    mag = np.abs(_image_dx) + np.abs(_image_dy)
    theta = np.arctan2(_image_dy, _image_dx)
    return mag, theta


Gaussian_dx_kernel = get_gaussian_kernel(1, 5, 'x')
Gaussian_dy_kernel = get_gaussian_kernel(1, 5, 'y')

plt.figure(figsize=(5, 5))
plt.subplot(121)
plt.imshow(Gaussian_dx_kernel)
plt.title('x derivative')
plt.subplot(122)
plt.imshow(Gaussian_dy_kernel)
plt.title('y derivative')
plt.show()

image_gaussian_dx = conv(srcImg, Gaussian_dx_kernel)
image_gaussian_dy = conv(srcImg, Gaussian_dy_kernel)
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(image_gaussian_dx/image_gaussian_dx.max())
plt.title('First order Gaussian Derivative x')
plt.subplot(122)
plt.imshow(image_gaussian_dy/image_gaussian_dy.max())
plt.title('First order Gaussian Derivative y')
plt.show()

mag, theta = get_gaussian_derivative(srcImg, 1)
plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(mag/mag.max())
plt.title('Magnitude')
plt.subplot(122)
plt.imshow(theta/theta.max())
plt.title('Theta')
plt.show()

mags = []
thetas = []
sigmas = [0.05, 1e-1, 1, 2, 5, 10]
for i in range(len(sigmas)):
    mag, theta = get_gaussian_derivative(srcImg, sigmas[i])
    mags.append(mag)
    thetas.append(theta)

# mag
plt.figure(figsize=(25, 25))
for i in range(6):
    mag = mags[i]
    plt.subplot(2, 3, i+1)
    plt.imshow(mag/mag.max())
    plt.title("Variance: " + str(sigmas[i]))

# theta
plt.figure(figsize=(25, 25))
for i in range(6):
    theta = thetas[i]
    plt.subplot(2, 3, i+1)
    plt.imshow(theta/theta.max())
    plt.title("Variance: " + str(sigmas[i]))
plt.show()
