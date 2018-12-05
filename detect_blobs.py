from skimage import io, color, img_as_float, draw, transform as tran
from math import sqrt, log
import numpy as np
import matplotlib.pyplot as plt
import cv2


# blog detection function
def blob_detect(image, sigma_min, sigma_max, ratio, thresh=0.05):
    img = color.rgb2gray(image)
    img = img_as_float(img)
    k = int(log(float(sigma_max) / sigma_min, ratio)) + 1
    sigmas = np.array([sigma_min * (ratio ** i) for i in range(k + 1)])
    imgs_filtered = [gauss_filter(img, sigma) for sigma in sigmas]
    imgs_dog = [(imgs_filtered[i] - imgs_filtered[i + 1]) for i in range(k)]
    # imgs_dog = [laplace_filter(imgs_filtered[i]) * sigmas[i] ** 2 for i in range(k)]
    img_pyramid = np.stack(imgs_dog, axis=-1) ** 2
    mask = np.zeros_like(img_pyramid)
    max_image = []
    for img_slice in range(img_pyramid.shape[-1]):
        max_image.append(max_filter(img_pyramid[:, :, img_slice], 3))
    max_image = np.stack(max_image, axis=-1)
    # mask[:, :, img_slice] = img_pyramid[:, :, img_slice] == max_image[img_slice]
    max_space = np.zeros_like(max_image)
    for ii in range(k):
        max_space[:, :, ii] = np.amax(max_image[:, :, max(ii-1, 0):min(ii+1, k)], axis=-1)
    mask = img_pyramid == max_space
    img_mask = mask & (img_pyramid > thresh)
    (cy, cx, sig) = np.nonzero(img_mask)
    for i in range(len(sig)):
        sig[i] = sigmas[sig[i]]
    return cy, cx, sig


def blob_detect_fast(image, sigma_min, sigma_max, scale, thresh=0.05):
    img = color.rgb2gray(image)
    img = img_as_float(img)
    k = int(log(float(sigma_max) / sigma_min, scale)) + 1
    scale_init = 1
    size_init = np.shape(img)
    scale_space = []
    sigmas = np.array([sigma_min * (scale ** i) for i in range(k + 1)])
    for i in range(k+1):
        scale_space.append(tran.rescale(img, 1 / (scale_init * scale ** i), anti_aliasing=True))

        # scale_space.append(cv2.resize(img, None, fx=1/(scale_init * scale ** i), fy=1/(scale_init * scale ** i), interpolation=cv2.INTER_AREA))
    imgs_filtered = [gauss_filter(img, sigma_min) for img in scale_space]
    imgs_log = []
    imgs_rescale = []
    for i in range(k+1):
        imgs_log.append(laplace_filter(imgs_filtered[i], sigma_min * scale ** i))
        imgs_rescale.append(tran.resize(imgs_log[i], size_init))
        # imgs_rescale.append(cv2.resize(imgs_log[i], (size_init[1], size_init[0]), interpolation=cv2.INTER_CUBIC))
    # imgs_dog = [(imgs_filtered[i] - imgs_filtered[i + 1]) for i in range(k)]
    # imgs_dog = [laplace_filter(imgs_filtered[i]) * sigma_min[i] ** 2 for i in range(k)]
    img_pyramid = np.stack(imgs_rescale, axis=-1) ** 2
    max_image = []
    for img_slice in range(k+1):
        max_image.append(max_filter(img_pyramid[:, :, img_slice], 3))
    max_image = np.stack(max_image, axis=-1)
    # mask[:, :, img_slice] = img_pyramid[:, :, img_slice] == max_image[img_slice]
    max_space = np.zeros_like(max_image)
    for ii in range(k+1):
        max_space[:, :, ii] = np.amax(max_image[:, :, max(ii-1, 0):min(ii+1, k)], axis=-1)
    mask = img_pyramid == max_image
    img_mask = mask & (img_pyramid > thresh)
    img_mask[0,:,:] = 0
    img_mask[-1,:,:] = 0
    img_mask[:,0,:] = 0
    img_mask[:,-1,:] = 0
    (cy, cx, sig) = np.nonzero(img_mask)
    for i in range(len(sig)):
        sig[i] = sigmas[sig[i]]
    return cy, cx, sig


def blob_detect_log(image, sigma_min, sigma_max, scale, thresh=0.05):
    img = color.rgb2gray(image)
    img = img_as_float(img)
    k = int(log(float(sigma_max) / sigma_min, scale)) + 1
    scale_init = 1
    size_init = np.shape(img)
    scale_space = []
    sigmas = np.array([sigma_min * (scale ** i) for i in range(k + 1)])
    for i in range(k+1):
        # scale_space.append(tran.rescale(img, 1 / (scale_init * scale ** i), anti_aliasing=True))

        scale_space.append(cv2.resize(img, None, fx=1/(scale_init * scale ** i), fy=1/(scale_init * scale ** i), interpolation=cv2.INTER_AREA))
    imgs_filtered = [log_filter(img, sigma_min) for img in scale_space]
    imgs_rescale = []
    for i in range(k+1):
        # imgs_rescale.append(tran.resize(imgs_filtered[i], size_init))
        imgs_rescale.append(cv2.resize(imgs_filtered[i], (size_init[1], size_init[0]), interpolation=cv2.INTER_CUBIC))
    img_pyramid = np.stack(imgs_rescale, axis=-1) ** 2
    max_image = []
    for img_slice in range(k+1):
        max_image.append(max_filter(img_pyramid[:, :, img_slice], 3))
    max_image = np.stack(max_image, axis=-1)
    # mask[:, :, img_slice] = img_pyramid[:, :, img_slice] == max_image[img_slice]
    max_space = np.zeros_like(max_image)
    for ii in range(k+1):
        max_space[:, :, ii] = np.amax(max_image[:, :, max(ii-1, 0):min(ii+1, k)], axis=-1)
    mask = img_pyramid == max_image
    img_mask = mask & (img_pyramid > thresh)
    img_mask[0,:,:] = 0
    img_mask[-1,:,:] = 0
    img_mask[:,0,:] = 0
    img_mask[:,-1,:] = 0
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
    h1 = h * (x*x + y*y - 2*std)/(std ** 2)
    h1sum = h1.sum()
    h_log = h1 - h1sum/size2
    img_filtered = conv2d(image, h_log, 'same', 'zero')
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
    # img_new = np.zeros_like(img)
    # img_new = np.zeros_like(img)
    # for row in range(0, h + 1):
    #     for col in range(0, w + 1):
    #         s2 = - min(-1, h - row - pad)
    #         value = kern * img[max(0, row - pad) : min(row + pad + 1, h + 1), max(0, col - pad):min(col + pad + 1, w + 1)]
    #         img_new[row, col] = min(1, max(0, value.sum()))
    return img_new


def laplace_filter(img, scale):
    kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_new = conv2d(img, kern, 'same', 'zero')
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
    file = 'TestImages4Project/butterfly.jpg'
    image = io.imread(file)
    # image = color.gray2rgb(image)
    sigma_min = 2
    sigma_max = 30
    ratio = 1.4788  # 10 scales, DoG ratio should be between sqrt(2) and 1.6
    thresh = 0.001
    scale = ratio
    (cy, cx, sig) = blob_detect_log(image, sigma_min, sigma_max, ratio, thresh)
    # image_gray = color.rgb2gray(image)
    # indices = blob_dog(image_gray, sigma_min, sigma_max, ratio, thresh)
    # cy = indices[:, 0]
    # cx = indices[:, 1]
    # sig = indices[:, 2]
    rad = sig.astype(float) * sqrt(2)
    plt.figure(1)
    plt.imshow(image)
    # new_image = np.pad(image, ((50, 50), (50, 50), (0, 0)), 'constant', constant_values=0)
    ax = plt.gcf().gca()
    for center_y, center_x, radius in zip(cy, cx, rad):
        circle = plt.Circle((center_x, center_y), radius, color='r', fill=0)
        ax.add_artist(circle)
    plt.show()

        # circy, circx = draw.circle_perimeter(center_y, center_x, radius)
        # circy += 50
        # circx += 50
        # new_image[circy, circx, :] = (255, 255, 20)

