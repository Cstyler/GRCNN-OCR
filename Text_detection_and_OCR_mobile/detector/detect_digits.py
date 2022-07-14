from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from SkynetCV import SkynetCV
from tensorflow.keras.models import Model
import math as m
import sys
# for gamma function, called
from scipy.special import gamma as tgamma
import os
import glob

from price_detector.detector.binarization import binarization
from price_detector.detector.box_utils import boxes_from_regions, \
    divide_boxes_that_contain_another_boxes, expand_boxes
from price_detector.detector.filters import filter_boxes_by_ar, filter_boxes_by_area, \
    filter_boxes_by_iou, filter_boxes_by_region_and_box_area, filter_boxes_by_same, \
    filter_boxes_by_same_digits, \
    filter_boxes_by_same_iou, filter_small_boxes
from price_detector.detector.utils import show_boxes_on_image
from tools.inference import infer_crossentropy_classification
import cv2

DIGITS_MAP = {10: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
              11: 11}

DIGIT_DEST_SHAPE = (32, 32)
BATCH_SIZE = 128
NO_OBJECT_CLASS = 0
BOX_EXPAND_SIZE = 1
MAX_INTENSITY = 255


def digit_classification_and_filter(boxes: np.ndarray,
                                    digit_classifier_model: Model,
                                    img: np.ndarray,
                                    scale_x: float, scale_y: float) -> Tuple[np.ndarray,
                                                                             np.ndarray]:
    batch_shape = DIGIT_DEST_SHAPE + (1,)
    w, h = DIGIT_DEST_SHAPE
    img_batch = np.array([crop_for_classifier(img, h, w,
                                              scale_x, scale_y, box, batch_shape)
                          for box in boxes])
    preds = digit_classifier_model.predict(img_batch, batch_size=BATCH_SIZE)
    boxes, classes = filter_boxes_by_noclass(boxes, preds)
    return boxes, classes


def crop_for_classifier(img, h, w, scale_x, scale_y, box, batch_shape):
    if scale_y != 1. or scale_x != 1.:
        box = box.copy()  # todo remove?
        box[[0, 2]] = (box[[0, 2]] * scale_x).astype(np.uint16)
        box[[1, 3]] = (box[[1, 3]] * scale_y).astype(np.uint16)
    xmin, ymin, xmax, ymax = box
    digit = img[ymin:ymax, xmin:xmax]
    digit = SkynetCV.resize(digit, w, h)
    digit = digit.reshape(batch_shape)
    return digit


def filter_boxes_by_noclass(boxes, preds):
    no_symbol_boxes = []
    classes = []
    for i, pred in enumerate(preds):
        pred_class = pred.argmax()
        classes.append(pred_class)
        if pred_class == NO_OBJECT_CLASS:
            no_symbol_boxes.append(i)
    boxes = np.delete(boxes, no_symbol_boxes, axis=0)
    classes = np.delete(classes, no_symbol_boxes, axis=0)
    return boxes, classes


def detect_digits(img: np.ndarray, digit_classifier_model: Model,
                  mser_params: Dict[str, Union[int, float]],
                  area_filter_params: Dict[str, Union[int, float]],
                  aspect_ratio_filter_params: Dict[str, float],
                  threshold_dict: Dict[str, float], box_expand_size: int,
                  params_dict: Optional[Dict[str, float]] = None, max_side: int = 120,
                  show: bool = False, img_save_list: Optional[List[str]] = None,
                  optimize: bool = False,
                  divide_algo_param_dict: Optional[Dict[str, float]] = None,
                  return_image: bool = False,
                  convert_to_gray: bool = True,
                  check_validity=False):
    # 1. Изменяем размер по большей стороне до max_side с сохранением соотношения сторон
    img_h, img_w, _ = img.shape
    if img_h > img_w:
        h = max_side
        w = int(h / img_h * img_w)
    else:
        w = max_side
        h = int(w / img_w * img_h)
    img_area = h * w

    # 2. Переводим изображение в оттенки серого
    if convert_to_gray:
        img_gray = SkynetCV.bgr2grayscale(img)
    else:
        img_gray = img

    if check_validity:

        valid_pricetag_classifier_model_path = '/home/ml/models/mlflow-artifact/35/f12c0b56a4a3461fb24474f4e512af89/artifacts/model_epoch-00310_loss-0.2973_val_loss-0.2809.tflite'
        pricetag_validity = infer_crossentropy_classification([img_gray],
                                                              valid_pricetag_classifier_model_path,
                                                              verbose=False,
                                                              keep_original_aspect_ratio=False)[0]
        if not pricetag_validity:
            if optimize:
                return [], None, None
            if return_image:
                return [], img_gray
            return []

    img_gray_resized = SkynetCV.resize(img_gray, w, h)

    if params_dict: binarization(img_gray_resized, params_dict, show)

    # 3. Запускаем конструктор MSER
    mser_params = mser_params.copy()
    min_area = mser_params["_min_area"]
    if isinstance(min_area, float):
        mser_params["_min_area"] = int(img_area * min_area)
    max_area = mser_params["_max_area"]
    if isinstance(max_area, float):
        mser_params["_max_area"] = int(img_area * max_area)
    mser = SkynetCV.MSER()
    mser.setParameters(mser_params["_min_area"], mser_params["_max_area"],
                       mser_params["_delta"], mser_params["_max_variation"])
    # 4. Определяем регионы MSER
    regions = mser.detect(img_gray_resized)

    # 5. Находим ограничивающие боксы для регионов
    boxes1 = boxes_from_regions(regions)

    # 6. Фильтруем боксы учитывая площадь регионов
    boxes2 = filter_boxes_by_region_and_box_area(boxes1, regions,
                                                 threshold_dict[
                                                     "region_box_area_ratio_threshold"])

    # 7. Фильтруем одинаковые боксы
    boxes3 = filter_boxes_by_same(boxes2)

    if show:
        write_to_file = img_save_list if img_save_list is None else img_save_list[0]
        show_boxes_on_image(boxes3, img_gray_resized.squeeze(),
                            write_to_file=write_to_file)

    if divide_algo_param_dict is None:
        boxes3_divide = boxes3
    else:
        show_divide = False
        img_show = img_gray_resized.squeeze(-1) if show_divide else None
        boxes3_divide = divide_boxes_that_contain_another_boxes(boxes3,
                                                                divide_algo_param_dict,
                                                                img_show)
    # 8. Фильтруем боксы по площади
    area_filter_params = area_filter_params.copy()
    min_area = area_filter_params["min_area"]
    if isinstance(min_area, float):
        area_filter_params["min_area"] = int(img_area * min_area)
    max_area = area_filter_params["max_area"]
    if isinstance(max_area, float):
        area_filter_params["max_area"] = int(img_area * max_area)
    boxes4 = filter_boxes_by_area(boxes3_divide, **area_filter_params)

    # 9. Фильтруем боксы по соотношению сторон
    boxes5 = filter_boxes_by_ar(boxes4, **aspect_ratio_filter_params)
    if not len(boxes5):
        if optimize:
            return boxes5, None, None
        return boxes5

    # 10. 11. Фильтрация похожих боксов в порядке убывания площади и возврат боксов в
    # порядке возрастания
    # площади
    boxes6 = filter_boxes_by_same_iou(boxes5, threshold_dict["same_iou_threshold"])

    # 12. Фильтруем боксы по iou
    boxes7 = filter_boxes_by_iou(boxes6, threshold_dict["iou_threshold"],
                                 threshold_dict["area_ratio_threshold"])

    if show:
        write_to_file = img_save_list if img_save_list is None else img_save_list[1]
        show_boxes_on_image(boxes7, img_gray_resized.squeeze(-1),
                            write_to_file=write_to_file)

    # 13. Увеличиваем размер боксов
    boxes8 = expand_boxes(boxes7, box_expand_size, h, w)

    # 14. Классификация боксов и удаление боксов, распознанных как фон
    # boxes9, classes = digit_classification_and_filter(boxes8, digit_classifier_model,
    #                                                   img_gray,
    #                                                   scale_x, scale_y)
    boxes9, classes = digit_classification_and_filter(boxes8, digit_classifier_model,
                                                      img_gray_resized,
                                                      1., 1.)

    if show:
        write_to_file = img_save_list if img_save_list is None else img_save_list[1]
        show_boxes_on_image(boxes9, img_gray_resized.squeeze(-1), classes,
                            write_to_file=write_to_file)

    # 14.1. Фильтрация похожих цифр
    boxes10, classes = filter_boxes_by_same_digits(boxes9, classes,
                                                   threshold_dict[
                                                       "same_digit_iou_threshold"])
    # 15. Фильтруем слишком маленькие боксы, не подходящие в качестве цифр
    boxes11 = filter_small_boxes(boxes10, img_area,
                                 threshold_dict["small_box_ratio_threshold"])

    if show and len(boxes11):
        print("boxes11", boxes11.shape)

        len_boxes = [len(_l) for _l in
                     (boxes1, boxes2, boxes3, boxes3_divide, boxes4, boxes5,
                      boxes6, boxes7, boxes9, boxes10, boxes11)]
        filter_names = ("start", "by_region_and_box_area", "by_same", "divide",
                        "by_area", "by_ar",
                        "by_same_iou", "by_iou", "classification_filter",
                        "by_same_digits", "small_boxes")
        print([f"{name}: {len_}" for name, len_ in zip(filter_names, len_boxes)])
        len_diffs = [len_boxes[i] - len_boxes[i + 1] for i in range(len(len_boxes) - 1)]
        print([f"{name}: {len_diff}" for name, len_diff in
               zip(filter_names[1:], len_diffs)])

    # Приведение к общему формату
    if len(boxes11):
        classes = np.array([DIGITS_MAP[cl] for cl in classes])
        boxes11 = np.insert(boxes11, 0, classes, axis=1)

    if show:
        show_boxes_on_image(boxes11, img_gray_resized.squeeze())
    if optimize:
        return boxes11, len(boxes8), len(boxes9)
    if return_image:
        return boxes11, img_gray
    return boxes11


def get_algo_params(version="v9", divide_algo: bool = True):
    params_dict = {"v1": {'area_ratio_threshold': 0.7740575, 'iou_threshold': 0.7784458,
                          'max_ar': 1.5038201, 'max_area': 0.4960071, 'min_ar': 0.2339092,
                          'min_area': 0.0004694, 'mser_delta': 3.0,
                          'mser_max_area': 0.5221199,
                          'mser_max_variation': 0.4237764, 'mser_min_area': 0.000888,
                          'region_box_area_ratio_threshold': 0.0060399,
                          'same_digit_iou_threshold': 0.6806329,
                          'same_iou_threshold': 0.7398102,
                          'small_box_ratio_threshold': 0.0004049, "box_expand_size": 1},
                   "v6": {'area_ratio_threshold': 0.7721431,
                          'iou_threshold': 0.7912409,
                          'max_ar': 1.5910618,
                          'max_area': 0.4189203,
                          'min_ar': 0.2515113,
                          'min_area': 0.0004641,
                          'mser_delta': 2.0,
                          'mser_max_area': 0.6484093,
                          'mser_max_variation': 0.5848091,
                          'mser_min_area': 0.0008331,
                          'region_box_area_ratio_threshold': 0.003961,
                          'same_digit_iou_threshold': 0.5645643,
                          'same_iou_threshold': 0.7355463,
                          'small_box_ratio_threshold': 0.0008433, "box_expand_size": 1},
                   'v9': {'area_ratio_threshold': 0.7162761,
                          'box_expand_size': 1,
                          'iou_threshold': 0.6974555,
                          'max_ar': 1.2347899,
                          'max_area': 0.2510429,
                          'min_ar': 0.2709671,
                          'min_area': 0.0009795,
                          'mser_delta': 7,
                          'mser_max_area': 0.5252633,
                          'mser_max_variation': 0.3063679,
                          'mser_min_area': 0.0009305,
                          'region_box_area_ratio_threshold': 0.004982,
                          'same_digit_iou_threshold': 0.5567707,
                          'same_iou_threshold': 0.7897797,
                          'small_box_ratio_threshold': 0.0006344},
                   'v10': {'area_ratio_threshold': 0.737283, 'box_expand_size': 1,
                           'iou_threshold': 0.606852, 'max_ar': 1.2132725,
                           'max_area': 0.2991612, 'min_ar': 0.2948103,
                           'min_area': 0.0009751, 'mser_delta': 6,
                           'mser_max_area': 0.5594896, 'mser_max_variation': 0.5577862,
                           'mser_min_area': 0.0001017,
                           'region_box_area_ratio_threshold': 0.0023073,
                           'same_digit_iou_threshold': 0.6298376,
                           'same_iou_threshold': 0.7585388,
                           'small_box_ratio_threshold': 0.0002829},
                   'v11': {'area_ratio_threshold': 0.6577636, 'box_expand_size': 1,
                           'iou_threshold': 0.7883265, 'max_ar': 1.2397613,
                           'max_area': 0.3879101, 'min_ar': 0.2998751,
                           'min_area': 0.0009997, 'mser_delta': 5,
                           'mser_max_area': 0.5611031, 'mser_max_variation': 0.2632004,
                           'mser_min_area': 0.000573,
                           'region_box_area_ratio_threshold': 0.0088387,
                           'same_digit_iou_threshold': 0.5505995,
                           'same_iou_threshold': 0.8775366,
                           'small_box_ratio_threshold': 0.0004641}
                   }
    best_params = params_dict[version]

    mser_params = dict(_delta=int(best_params["mser_delta"]),
                       _max_variation=best_params["mser_max_variation"],
                       _min_area=best_params["mser_min_area"],
                       _max_area=best_params["mser_max_area"])

    area_filter_params = dict(min_area=best_params["min_area"],
                              max_area=best_params["max_area"])

    aspect_ratio_filter_params = dict(min_ar=best_params["min_ar"],
                                      max_ar=best_params["max_ar"])

    box_expand_size = best_params["box_expand_size"]

    threshold_dict = best_params
    if divide_algo:
        divide_algo_param_dict = dict(
            x_coord_rel_tol=.01, y_coord_rel_tol=.05,
            min_width_ratio=.25, max_width_ratio=.55,
            min_height_ratio=.8, max_height_ratio=1.05,
            free_space_width_for_one_box_left_thr=.5,
            free_space_width_for_one_box_right_thr=1.5,
            free_space_width_for_two_boxes_right_thr=2.5,
            min_box_dist=.1,
            max_box_dist=1.5
        )
    else:
        divide_algo_param_dict = None
    return area_filter_params, aspect_ratio_filter_params, divide_algo_param_dict, \
           mser_params, threshold_dict, box_expand_size


def detect_blur_fft(image, size=60, thresh=10):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    # zero-out the center of the FFT shift
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean


def detect_glare(image, thresh=0.13):
    image = image[image.shape[0] // 2 - image.shape[0] // 5:, image.shape[1] // 2 - image.shape[1] // 5:]
    mask = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)[1]
    return (mask / 255).sum() / mask.size


def AGGDfit(structdis):
    # AGGD fit model, takes input as the MSCN Image / Pair-wise Product
    # variables to count positive pixels / negative pixels and their squared sum
    poscount = 0
    negcount = 0
    possqsum = 0
    negsqsum = 0
    abssum = 0

    poscount = len(structdis[structdis > 0])  # number of positive pixels
    negcount = len(structdis[structdis < 0])  # number of negative pixels

    # calculate squared sum of positive pixels and negative pixels
    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))

    # absolute squared sum
    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    # calculate left sigma variance and right sigma variance
    lsigma_best = np.sqrt((negsqsum / negcount))
    rsigma_best = np.sqrt((possqsum / poscount))

    gammahat = lsigma_best / rsigma_best

    # total number of pixels - totalcount
    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum / totalcount, 2) / ((negsqsum + possqsum) / totalcount)
    rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1) / (m.pow(m.pow(gammahat, 2) + 1, 2))

    prevgamma = 0
    prevdiff = 1e10
    sampling = 0.001
    gam = 0.2

    # vectorized function call for best fitting parameters
    vectfunc = np.vectorize(func, otypes=[np.float], cache=False)

    # calculate best fit params
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best]


def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while (gam < 10):
        r_gam = tgamma(2 / gam) * tgamma(2 / gam) / (tgamma(1 / gam) * tgamma(3 / gam))
        diff = abs(r_gam - rhatnorm)
        if (diff > prevdiff):
            break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best


def compute_features(img):
    scalenum = 2
    feat = []
    # make a copy of the image
    im_original = img.copy()
    feat.append(detect_blur_fft(img))
    feat.append(detect_glare(img))

    # scale the images twice
    for itr_scale in range(scalenum):
        im = im_original.copy()
        # normalize the image
        im = im / 255.0

        # calculating MSCN coefficients
        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166)
        sigma = (sigma - mu_sq) ** 0.5

        # structdis is the MSCN image
        structdis = im - mu
        structdis /= (sigma + 1.0 / 255)

        # calculate best fitted parameters from MSCN image
        best_fit_params = AGGDfit(structdis)
        # unwrap the best fit parameters
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best = best_fit_params[2]

        # append the best fit parameters for MSCN image
        feat.append(gamma_best)
        feat.append((lsigma_best * lsigma_best + rsigma_best * rsigma_best) / 2)

        # shifting indices for creating pair-wise products
        shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]  # H V D1 D2

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift - 1]  # shifting index

            # create transformation matrix for warpAffine function
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))

            Shifted_new_structdis = ShiftArr
            Shifted_new_structdis = Shifted_new_structdis * structdis
            # shifted_new_structdis is the pairwise product
            # best fit the pairwise product
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best = best_fit_params[2]

            constant = m.pow(tgamma(1 / gamma_best), 0.5) / m.pow(tgamma(3 / gamma_best), 0.5)
            meanparam = (rsigma_best - lsigma_best) * (tgamma(2 / gamma_best) / tgamma(1 / gamma_best)) * constant

            # append the best fit calculated parameters
            feat.append(gamma_best)  # gamma best
            feat.append(meanparam)  # mean shape
            feat.append(m.pow(lsigma_best, 2))  # left variance square
            feat.append(m.pow(rsigma_best, 2))  # right variance square

        # resize the image on next iteration
        im_original = cv2.resize(im_original, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return feat