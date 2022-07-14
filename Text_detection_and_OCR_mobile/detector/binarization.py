import cv2
import numpy as np
from skimage.filters import threshold_niblack

from price_detector.detector.utils import show_img


def binarization(img_gray, params, show: bool = False):
    window_size = params["window_size"]
    k = params["k"]
    thresh_niblack: np.ndarray = threshold_niblack(img_gray, window_size, k)

    binary_niblack = (img_gray > thresh_niblack).astype(np.uint8)
    ksize = params["open_ksize"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    opening = cv2.morphologyEx(binary_niblack, cv2.MORPH_OPEN, kernel)
    if show:
        print(binary_niblack.shape)
        show_img(binary_niblack)
        show_img(opening)
    return binary_niblack * 255
