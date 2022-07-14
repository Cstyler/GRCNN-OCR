import cv2
import numpy as np
from scipy.fftpack import dct, idct
import math
import os, glob
from matplotlib import pyplot as plt


def dct_2d(img):
    return dct(dct(img, 1, axis=0), 1, axis=1)


def idct_2d(img):
    return idct(idct(img, 1, axis=0), 1, axis=1)


def optimize(original_img, J, I):
    gain = J.mean(axis=(0, 1)) / I.mean(axis=(0, 1))
    offset = (J - gain * I)

    offset = cv2.resize(offset, (original_img.shape[1], original_img.shape[0]))

    original_img *= gain
    original_img += offset
    original_img = original_img.clip(0, 255).astype(np.uint8)

    return original_img


def enhance(img, lamb=.01, cs=1.5, resize_factor=None, resize_size=None):
    img_f = img.astype(float)

    if resize_factor is None and resize_size is None:
        resized = img_f.copy()
    elif resize_size is not None:
        resized = cv2.resize(img_f, (resize_size, resize_size)).astype(float)
    elif resize_factor is not None:
        resized = cv2.resize(img_f, None, fx=resize_factor, fy=resize_factor).astype(float)
    resized_shape = resized.shape

    img_fft = dct_2d(resized)

    all_255 = np.full(resized_shape, 255, dtype=np.float)
    all_255_fft = dct_2d(all_255)

    # prepare sx^2 + sy^2
    indices = np.indices((resized_shape[0], resized_shape[1])).astype(np.float)

    norm_x = (math.pi ** 2) / ((resized_shape[1] - 1) ** 2)
    norm_y = (math.pi ** 2) / ((resized_shape[0] - 1) ** 2)

    sx2_plus_sy2 = norm_x * np.square(indices[1, :, :]) + norm_y * np.square(indices[0, :, :])

    # print(resized.ndim)
    if resized.ndim == 3:
        sx2_plus_sy2 = np.repeat(sx2_plus_sy2[:, :, np.newaxis], resized_shape[2], axis=2)

    pi_square_x4 = 4 * (math.pi ** 2)
    wx2_plus_wy2 = pi_square_x4 * sx2_plus_sy2

    fft_solution = (lamb * all_255_fft + cs * wx2_plus_wy2 * img_fft) / (lamb + wx2_plus_wy2)

    result = idct_2d(fft_solution) / (4 * resized_shape[0] * resized_shape[1])
    result = result.clip(0, 255)

    opt_img = optimize(img_f, result, resized)

    return opt_img


def equalize_hist(img):
    img_f = img.astype(float)
    out_img = 255. * (img_f - img_f.min(axis=(0, 1))) / (img_f.max(axis=(0, 1)) - img_f.min(axis=(0, 1)))
    out_img = out_img.clip(0, 255).astype(np.uint8)
    return out_img


def increase_contrast_hsv(img, saturate_threshold=0.05, value_factor=2.):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    hsv[..., 1] *= value_factor
    hsv[..., 1] = hsv[..., 1].clip(0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def simplest_color_balance(img, saturate=0.1):
    '''
    :param saturate: [0,100]
    '''

    flat_BGR = img.reshape((3, -1), order='F')
    sorted_BGR = np.sort(flat_BGR, axis=-1)

    idx_min = int(sorted_BGR.shape[1] * saturate / 100.)
    idx_max = int(sorted_BGR.shape[1] - idx_min - 1)

    out = []
    for i, channel in enumerate(flat_BGR):
        v_min = sorted_BGR[i, idx_min]
        v_max = sorted_BGR[i, idx_max]
        out.append(255 * (channel.astype(float).clip(v_min, v_max) - v_min) / (v_max - v_min))

    out_img = np.reshape(out, img.shape, 'F').clip(0, 255).astype(np.uint8)

    return out_img


def adjust_gamma(image, gamma=0.65):
    '''
    :param gamma: [0.1,10]
    '''
    invGamma = 1.0 / gamma
    table = ((np.linspace(0, 1, 256) ** invGamma) * 255).astype(np.uint8)
    return cv2.LUT(image, table)


def seamless(img, gray=False):
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.tile(np.expand_dims(img, -1), 3)

    white_bg = 255 * np.ones(img.shape, img.dtype)
    mask = 255 * np.ones(img.shape, img.dtype)

    height, width = img.shape[:2]
    center = (int(width / 2), int(height / 2))

    normal_clone = cv2.seamlessClone(img, white_bg, mask, center, cv2.NORMAL_CLONE)
    mixed_clone = cv2.seamlessClone(img, white_bg, mask, center, cv2.MIXED_CLONE)

    return normal_clone, mixed_clone


def imshow(img, title):
    plt.title(title)
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[..., ::-1])
    plt.show()


if __name__ == '__main__':

    filepaths = glob.glob('/home/ml/datasets/schwarzkopf-retail/price_tags/schwarzkopf_pt_labeling/*')
    x = [cv2.imread(f) for f in filepaths]

    for img in x[:20]:
        imshow(img, 'img')

        #     seamless
        #     normal_clone, mixed_clone = seamless(img, gray=True)
        #     imshow(mixed_clone, 'mixed_clone_gray')
        #     equalize_hist_img = equalize_hist(mixed_clone)
        #     imshow(equalize_hist_img, 'equalize_hist_img')

        #     poisson
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     enhanced_img = enhance(gray, lamb=0.01, cs=1.5, resize_size=100)
        enhanced_img = enhance(gray, lamb=0.1, cs=2.5, resize_size=100)
        imshow(enhanced_img, 'gray enhanced_img')
