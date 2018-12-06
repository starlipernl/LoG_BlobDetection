"""""
# Nathan Starliper
# ECE 558 Final Project
Scale invariant blob detection using LoG - This code detects SIFT blobs in a fast efficient manner using
the laplacian of gaussian of an image pyramid to detect blobs at varying scales
"""

from skimage import io, color, img_as_float, transform as tran
from math import sqrt, log
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.simplefilter('ignore')


# Fastest and main implementation using LoG and image scaling with constant filter size
def blob_detect(image, sigma_min, sigma_max, scale, thresh):
    # convert image to grayscale float [0,1]
    img = color.rgb2gray(image)
    img = img_as_float(img)
    # integer number of scales between minimum and maximum sigma based on scale ratio
    k = int(log(float(sigma_max) / sigma_min, scale)) + 1
    scale_init = 1
    size_init = np.shape(img)
    # list of sigma scales
    sigmas = np.array([sigma_min * (scale ** i) for i in range(k + 1)])
    # create the image pyramid scale space resizing images
    scale_space = []
    for i in range(k+1):
        scale_space.append(tran.rescale(img, 1 / (scale_init * scale ** i), order=3, clip=False))
    # filter scaled images using LoG filter
    imgs_filtered = [log_filter(img, sigma_min) for img in scale_space]
    # rescale images to original scale
    imgs_rescale = []
    for i in range(k+1):
        imgs_rescale.append(tran.resize(imgs_filtered[i], size_init, order=3, clip=False))
    img_pyramid = np.stack(imgs_rescale, axis=-1) ** 2
    max_image = []
    # max filter images within scales with 3x3 filter
    for img_slice in range(k+1):
        max_image.append(max_filter(img_pyramid[:, :, img_slice], 3))
    max_image = np.stack(max_image, axis=-1)
    max_space = np.zeros_like(max_image)
    # max filter across scales with depth of 3
    for ii in range(k+1):
        max_space[:, :, ii] = np.amax(max_image[:, :, max(ii-1, 0):min(ii+1, k)], axis=-1)
    # mask scale space with max space and find coordinates of each blob
    max_space *= img_pyramid == max_space
    (cy, cx, sig) = np.where(max_space >= thresh)
    # convert coordinate of 3rd dimension to corresponding sigma value
    for i in range(len(sig)):
        sig[i] = sigmas[sig[i]]
    return cy, cx, sig


# blog detection function DoG approach, scaling filter
def blob_detect_slow(image, sigma_min, sigma_max, ratio, thresh):
    img = color.rgb2gray(image)
    img = img_as_float(img)
    k = int(log(float(sigma_max) / sigma_min, ratio)) + 1
    sigmas = np.array([sigma_min * (ratio ** i) for i in range(k + 1)])
    imgs_filtered = [log_filter(img, sigma) for sigma in sigmas]
    img_pyramid = np.stack(imgs_filtered, axis=-1) ** 2
    max_image = []
    for img_slice in range(img_pyramid.shape[-1]):
        max_image.append(max_filter(img_pyramid[:, :, img_slice], 3))
    max_image = np.stack(max_image, axis=-1)
    max_space = np.zeros_like(max_image)
    for ii in range(k):
        max_space[:, :, ii] = np.amax(max_image[:, :, max(ii-1, 0):min(ii+1, k)], axis=-1)
    mask = img_pyramid == max_space
    img_mask = mask & (img_pyramid > thresh)
    (cy, cx, sig) = np.nonzero(img_mask)
    for i in range(len(sig)):
        sig[i] = sigmas[sig[i]]
    return cy, cx, sig


# 3x3 gauss blurring filter with scale sigma
def gauss_filter(image, sigma):
    # round to odd integer size
    size = np.ceil(3 * sigma) // 2 * 2
    y, x = np.mgrid[-size:size+1, -size:size+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    hsum = h.sum()
    if hsum != 0:
        h /= hsum
    # filter image using gaussian filter h
    img_filtered = conv2d(image, h, 'same', 'zero')
    return img_filtered


# 3x3 gauss blurring filter with scale sigma
def log_filter(image, sigma):
    # round to odd integer size
    std = sigma ** 2
    size = np.ceil(3 * sigma) // 2 * 2
    size2 = np.float64((size * 2 + 1) ** 2)
    y, x = np.mgrid[-size:size+1, -size:size+1]
    h = np.exp(-(x*x + y*y) / (2.*std))
    hsum = h.sum()
    if hsum != 0:
        h /= hsum
    # laplacian of gaussian
    h1 = h * (x*x + y*y - 2*std)/(std ** 2)
    h1sum = h1.sum()
    h_log = std * (h1 - h1sum/size2)
    # filter image
    img_filtered = conv2d(image, h_log, 'same', 'copy')
    return img_filtered


# actual 2d convolution implementation function
def conv2d(img, kern, opt_shape, opt_pad):
    # pad the image according to the padding option input
    (m, n) = np.shape(kern)
    pad = int((m-1)/2)
    (h, w) = np.shape(img)
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
        (yr, xr) = np.shape(img_pad)
        (xi, yi) = (0, 0)
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
                    val = img_pad[i - pad:, j - pad:] * kern
                    img_new[i][j] = val.sum()
                else:
                    val = img_pad[i - pad:, j - pad:j+pad+1] * kern
                    img_new[i][j] = val.sum()
            elif j == w-1:
                val = img_pad[i - pad:i + pad + 1, j - pad:] * kern
                img_new[i][j] = val.sum()
            else:
                val = img_pad[i-pad:i+pad+1, j-pad:j+pad+1] * kern
                img_new[i][j] = val.sum()
    # remove padding depending on shape options input
    if opt_shape == 'same':
        img_new = img_new[pad:-pad, pad:-pad]
    elif opt_shape == 'full':
        img_new = img_new[pad:-pad, pad:-pad]
    elif opt_shape == 'valid':
        img_new = img_pad[pad+1:-(pad+1), pad+1:-(pad+1)]
    return img_new


# function for implementing a maximum filter
def max_filter(image, size):
    pad_width = (size-1)//2
    image_pad = np.pad(image, ((pad_width, pad_width),), 'constant', constant_values=((0, 0),))
    max_image = np.zeros_like(image)
    for x in range(image.shape[1]):  # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # assign max value of local region to pixel
            max_image[y, x] = np.amax(image_pad[y:y + size, x:x + size])
    return max_image


if __name__ == '__main__':
    # load images and initialize parameters
    file = 'butterfly.jpg'
    image = io.imread(file)
    sigma_min = 2
    sigma_max = 50
    ratio = 1.2394  # 15 total scales
    thresh = 0.06
    t1 = time.time()
    # run blob detector function
    (cy, cx, sig) = blob_detect(image, sigma_min, sigma_max, ratio, thresh)
    rad = sig.astype(float) * sqrt(2)  # convert sigma value to radius r = sqrt(2)*sigma
    t2 = time.time()
    print('Total Blob Detection Time:')
    print(t2-t1)
    plt.figure(1)
    plt.imshow(image, cmap='gray')
    ax = plt.gcf().gca()
    # plot circles to mark blobs
    for center_y, center_x, radius in zip(cy, cx, rad):
        circle = plt.Circle((center_x, center_y), radius, color='r', fill=0)
        ax.add_artist(circle)
    plt.show()
