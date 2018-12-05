# Author: Nathan Starliper
# ECE 542 HW02 Problem 3
# 10/2/2018

# This script implements a 2D convolution function as specified in problem 3.

from skimage import io, color, exposure
import numpy as np
from matplotlib import pyplot as plt


# actual 2d convolution implementation function
def conv2d(img, kern, opt_shape, opt_pad):
    # pad the image according to the padding option input
    (m, n) = np.shape(kern)
    pad = int((m-1)/2)
    # pad according to padding option input
    if opt_pad == 'zero':
        # make all border pixels black
        img_pad = np.pad(img, pad, 'constant', constant_values=0)
    elif opt_pad == 'wrap':
        # wrap the pixels to the other side
        img_pad = np.pad(img, pad, 'wrap')
    elif opt_pad == 'copy':
        img_pad = np.pad(img, pad, 'edge')
    elif opt_pad == 'reflect':
        img_pad = np.pad(img, pad, 'reflect')
    else:
        print('Incorrect padding option, please specify: zero, wrap, copy, reflect.')
        return 0
    # if full shape, pad with zeros again (these will be removed later)
    if opt_shape == 'full':
        img_pad = np.pad(img_pad, pad, 'constant', constant_values=0)
        # (yr, xr) = np.shape(img_pad)
        # (xi, yi) = (0, 0)
    # set sliding convolution window limits
    (h, w) = np.shape(img_pad)
    img_new = np.zeros((h, w))
    yi = pad
    yf = h - pad
    xi = pad
    xf = w - pad

    for i in range(yi, yf):
        for j in range(xi, xf):
            # different limits needed for boundary condition
            if i == h-1:
                if j == w-1:
                    img_new[i][j] = np.sum(img_pad[i - pad:, j - pad:] * kern)
                else:
                    img_new[i][j] = np.sum(img_pad[i - pad:, j - pad:j+pad+1] * kern)
            elif j == w-1:
                img_new[i][j] = np.sum(img_pad[i - pad:i + pad + 1, j - pad:] * kern)
            else:
                img_new[i][j] = np.sum(img_pad[i-pad:i+pad+1, j-pad:j+pad+1]*kern)
    # remove padding depending on shape options input
    if opt_shape == 'same':
        img_new = img_new[pad:-pad, pad:-pad]
    elif opt_shape == 'full':
        img_new = img_new[pad:-pad, pad:-pad]
    elif opt_shape == 'valid':
        img_new = img_pad[pad+1:-(pad+1), pad+1:-(pad+1)]
    return img_new


# function that implements the sobel kernel. This function outputs the x, y , and total filtered image
def sobelflt(img, opt_shape, opt_pad):
    kernx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kerny = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_newx = conv2d(img, kernx, opt_shape, opt_pad)
    img_new_equalizedx = exposure.equalize_adapthist(img_newx / np.max(np.abs(img_newx)), clip_limit=0.03)
    img_newy = conv2d(img, kerny, opt_shape, opt_pad)
    img_new_equalizedy = exposure.equalize_adapthist(img_newy / np.max(np.abs(img_newy)), clip_limit=0.03)
    img_new = np.sqrt((img_newx * img_newx) + (img_newy * img_newy))
    img_new_equalized = exposure.equalize_adapthist(img_new / np.max(np.abs(img_new)), clip_limit=0.03)
    return img_new_equalizedx, img_new_equalizedy, img_new_equalized


# function to implement the laplace filter
def laplaceflt(img, opt_shape, opt_pad):
    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_new = conv2d(img, kern, opt_shape, opt_pad)
    img_new_equalized = exposure.equalize_adapthist(img_new / np.max(np.abs(img_new)), clip_limit=0.03)
    return img_new_equalized

if __name__ == '__main__':
    # load image, and initialize variables/parameters
    image = io.imread('C:/Users/starl/Documents/ECE558_Digital_Imaging_Systems/HW2/lena-gray.bmp')
    img = color.rgb2gray(image)
    # apply Histogram Equalization
    image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
    opt_shape = 'same'
    opt_pad = 'copy'

    # Perform sobel filtering and save the x filtered, y filtered, and total sobel filtered image
    img_sobelx, img_sobely, img_sobel = sobelflt(image_equalized, opt_shape, opt_pad)
    plt.imshow(img_sobel, cmap=plt.cm.gray)
    plt.show()
    io.imsave('C:/Users/starl/Documents/ECE558_Digital_Imaging_Systems/HW2/img_sobelx.png', img_sobelx)
    io.imsave('C:/Users/starl/Documents/ECE558_Digital_Imaging_Systems/HW2/img_sobely.png', img_sobely)
    io.imsave('C:/Users/starl/Documents/ECE558_Digital_Imaging_Systems/HW2/img_sobel.png', img_sobel)

    # Perform laplacian filtering and save the filtered image
    img_laplace = laplaceflt(image_equalized, opt_shape, opt_pad)
    plt.imshow(img_laplace, cmap=plt.cm.gray)
    plt.show()
    io.imsave('C:/Users/starl/Documents/ECE558_Digital_Imaging_Systems/HW2/img_laplace.png', img_laplace)

