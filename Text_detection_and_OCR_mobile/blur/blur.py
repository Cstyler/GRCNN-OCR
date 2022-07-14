from typing import Optional, Tuple

from SkynetCV import SkynetCV
import cv2
# import matplotlib.pyplot as plt
import numpy as np

from price_detector.data_processing.utils import read_pickle_local
from .hwt_blur import hwt_blur_detect

ksize_laplacian = 3
ksize_median = 3


def laplacian_stats(img: np.ndarray,
                    dsize: Optional[int] = None) -> Tuple[float, float]:
    if len(img.shape) != 2:
        img = SkynetCV.bgr2grayscale(img)
    if dsize is not None:
        img = SkynetCV.resize(img, dsize, dsize)
    img = cv2.medianBlur(img, ksize_median)
    laplacian = cv2.Laplacian(img, -1, ksize=ksize_laplacian)
    return laplacian.mean(), laplacian.std()


def calc_blur_metrics(img_list_filename: str, box_arrays_filename: str,
                      dsize: Optional[int] = None):
    img_arr = read_pickle_local(img_list_filename)
    boxes_arr = read_pickle_local(box_arrays_filename)
    blur_means = []
    blur_stds = []
    for img, boxes in zip(img_arr, boxes_arr):
        means_batch, means_batch_mean, std_batch, std_batch_mean = calc_mean_blur_metrics(
            boxes, dsize, img)
        if len(means_batch):
            blur_means.append(means_batch_mean)
        if len(std_batch):
            blur_stds.append(std_batch_mean)
    return np.asarray(blur_means), np.asarray(blur_stds)


def calc_mean_blur_metrics(boxes, dsize, img):
    means_batch = []
    std_batch = []
    for box in boxes:
        _, xmin, ymin, xmax, ymax = box
        crop = img[ymin:ymax, xmin:xmax]
        # plt.imshow(crop)
        # plt.show()
        # plt.close()
        w, h, _ = crop.shape
        if w * h > 1:
            # print(crop.shape)
            mean, std = laplacian_stats(crop, dsize)
            means_batch.append(mean)
            std_batch.append(std)
    means_batch_mean = np.mean(means_batch)
    std_batch_mean = np.mean(std_batch)
    return means_batch, means_batch_mean, std_batch, std_batch_mean


def calc_haar_mean_metrics(img, dsize, threshold):
    # img = SkynetCV.resize(np.expand_dims(img, -1), dsize, dsize).squeeze()
    per, be = hwt_blur_detect(img, threshold)
    return per, be
