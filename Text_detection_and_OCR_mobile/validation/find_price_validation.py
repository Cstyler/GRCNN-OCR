import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from price_detector.data_processing.utils import read_array_local, read_pickle_local, \
    write_pickle_local
from price_detector.detector.detect_digits import detect_digits
from price_detector.detector.utils import show_boxes_on_image
from price_detector.enhancing.document_scanner.scan import DocScanner
from price_detector.enhancing.enhancing import enhance as enhance_fft
from price_detector.enhancing.noteshrink import enhance_noteshrink
from price_detector.find_price import find_price, dist
from price_detector.recognizer import PriceRecognizer

TQDM_SMOOTH = .01

MAIN_DIR = Path('..')
MODELS_DIR = MAIN_DIR / 'models'
DATASETS_DIR = MAIN_DIR / 'datasets'
PNG_FORMAT = "%s.png"

FilterCallable = Callable[[float], bool]


def calc_val_metrics(img_list_filename: str,
                     prices_filename: str,
                     algo_params: dict,
                     version: str = "v1",
                     ids: Optional[list] = None,
                     save_pickle: Optional[str] = None,
                     gpu=True,
                     tqdm_enable: bool = True,
                     show: bool = False,
                     stage2: bool = False):
    prices_array = read_array_local(prices_filename)
    img_list = read_array_local(img_list_filename)
    price_recognizer = PriceRecognizer(version, gpu)
    if stage2:
        return calc_val_metrics_iter_stage2(img_list, prices_array, price_recognizer,
                                            algo_params, ids, save_pickle, show=show,
                                            tqdm_enable=tqdm_enable)
    else:
        return calc_val_metrics_iter(img_list, prices_array, price_recognizer,
                                     algo_params, ids, save_pickle, show=show,
                                     tqdm_enable=tqdm_enable)


def calc_val_metrics_iter_stage2(img_list: np.ndarray, prices_array: np.ndarray,
                                 price_recognizer: PriceRecognizer,
                                 algo_params: Dict[str, Union[int, float]],
                                 ids: Optional[List[int]] = None,
                                 save_pickle: Optional[str] = None,
                                 save_pickle2=None, save_pickle3=None,
                                 show: bool = False, tqdm_enable: bool = False):
    soft_metrics, hard_metrics, \
    hard_det_metrics, soft_det_metrics, detections = [], [], [], [], []
    res_ids = []
    try:
        img_boxes_tuples = read_pickle_local(save_pickle)
        if save_pickle2 is not None and save_pickle3 is not None:
            pred_boxes_list2 = read_pickle_local(save_pickle2)
            pred_boxes_list3 = read_pickle_local(save_pickle3)
            img_boxes_tuples.extend(pred_boxes_list2)
            img_boxes_tuples.extend(pred_boxes_list3)
    except:
        iter = tqdm(img_list, total=len(img_list), disable=not tqdm_enable,
                    smoothing=TQDM_SMOOTH)
        img_boxes_tuples = [price_recognizer.detect(img, return_image=True, show=show)
                            for img in iter]
        write_pickle_local(save_pickle, img_boxes_tuples)
    iter = enumerate(zip(img_list, prices_array, img_boxes_tuples))
    for i, (img, price_true, (pred_boxes, img_resized)) in tqdm(iter, total=len(img_list),
                                                                disable=not tqdm_enable,
                                                                smoothing=TQDM_SMOOTH):
        if ids is not None and i not in ids:
            continue
        price_pred, price_boxes, \
        rub_bboxes, kop_bboxes \
            = price_recognizer.recognize_price_from_boxes(img_resized,
                                                          pred_boxes, False,
                                                          algo_params,
                                                          stage2=True)
        price_true = round(float(price_true), 2)
        price_pred = round(float(price_pred), 2)
        soft_metric = int(price_pred) == int(price_true)
        hard_metric = price_true == price_pred
        if len(pred_boxes):
            digits_pred = [str(x) for x in pred_boxes[:, 0]]
            num_str = str(price_true).replace('.', '')
            good_detection_flag = all(dt in digits_pred for dt in num_str)
        else:
            good_detection_flag = False
        if show and not hard_metric and good_detection_flag:
            price_recognizer.recognize_price_from_boxes(img_resized, pred_boxes, True,
                                                        algo_params, stage2=True)
            print("rub_bboxes", rub_bboxes, "kop_bboxes", kop_bboxes)
            print("price pred:", price_pred,
                  "price true:", price_true,
                  "Rub metrics:", soft_metric, "Rub+kop metric:", hard_metric)
            print("pred boxes:", pred_boxes)
            resize_boxes(img, pred_boxes, price_recognizer.max_side)
            show_boxes_on_image(pred_boxes, img)
        # price_recognizer.detect(img, True)
        soft_metrics.append(soft_metric)
        hard_metrics.append(hard_metric)
        if good_detection_flag:
            hard_det_metrics.append(hard_metric)
            soft_det_metrics.append(soft_metric)

    if show: print("Res ids:", res_ids)
    return np.mean(soft_metrics), np.mean(hard_metrics), \
           np.mean(soft_det_metrics), np.mean(hard_det_metrics), \
           len(soft_det_metrics)


def calc_val_metrics_iter(img_list: np.ndarray, prices_array: np.ndarray,
                          price_recognizer: PriceRecognizer,
                          algo_params: Dict[str, Union[int, float]],
                          ids: Optional[List[int]] = None,
                          save_pickle: Optional[str] = None,
                          show: bool = False, tqdm_enable: bool = False):
    soft_metrics, hard_metrics, \
    hard_det_metrics, soft_det_metrics, detections = [], [], [], [], []
    res_ids = []
    try:
        pred_boxes_list = read_pickle_local(save_pickle)
    except:
        iter = tqdm(img_list,
                    total=len(img_list), disable=not tqdm_enable, smoothing=TQDM_SMOOTH)
        pred_boxes_list = [price_recognizer.detect(img) for img in iter]
        write_pickle_local(save_pickle, pred_boxes_list)
    iter = enumerate(zip(img_list, prices_array, pred_boxes_list))
    for i, (img, price_true, pred_boxes) in tqdm(iter, total=len(img_list),
                                                 disable=not tqdm_enable,
                                                 smoothing=TQDM_SMOOTH):
        if ids is not None and i not in ids:
            continue
        price_pred, price_boxes, \
        rub_bboxes, \
        kop_bboxes = price_recognizer.recognize_price_from_boxes(img, pred_boxes,
                                                                 False, algo_params)
        price_true = round(float(price_true), 2)
        price_pred = round(float(price_pred), 2)
        soft_metric = int(price_pred) == int(price_true)
        hard_metric = price_true == price_pred
        if len(pred_boxes):
            digits_pred = [str(x) for x in pred_boxes[:, 0]]
            num_str = str(price_true).replace('.', '')
            good_detection_flag = all(dt in digits_pred for dt in num_str)
        else:
            good_detection_flag = False
        if show and not hard_metric and good_detection_flag:
            price_recognizer.recognize_price_from_boxes(img, pred_boxes,
                                                        True, algo_params)
            print("rub_bboxes", rub_bboxes, "kop_bboxes", kop_bboxes)
            print("price pred:", price_pred,
                  "price true:", price_true,
                  "Rub metrics:", soft_metric, "Rub+kop metric:", hard_metric)
            print("pred boxes:", pred_boxes)
            resize_boxes(img, pred_boxes, price_recognizer.max_side)
            show_boxes_on_image(pred_boxes, img)
        # price_recognizer.detect(img, True)
        soft_metrics.append(soft_metric)
        hard_metrics.append(hard_metric)
        if good_detection_flag:
            hard_det_metrics.append(hard_metric)
            soft_det_metrics.append(soft_metric)

    if show: print("Res ids:", res_ids)
    return np.mean(soft_metrics), np.mean(hard_metrics), \
           np.mean(soft_det_metrics), np.mean(hard_det_metrics), \
           len(soft_det_metrics)


def resize_boxes(img: np.ndarray, pred_boxes: np.ndarray, max_side: int):
    img_h, img_w, _ = img.shape
    if img_h > img_w:
        h = max_side
        w = int(h / img_h * img_w)
    else:
        w = max_side
        h = int(w / img_w * img_h)
    if len(pred_boxes):
        pred_boxes[:, [1, 3]] = (pred_boxes[:, [1, 3]] * img_w / w).astype(np.uint16)
        pred_boxes[:, [2, 4]] = (pred_boxes[:, [2, 4]] * img_h / h).astype(np.uint16)


def calc_val_metrics_blur(img_list_filename: str,
                          prices_filename: str,
                          boxes_array_filename: str,
                          algo_params: dict,
                          blur_params: Optional[Dict[str, float]] = None,
                          ids: list = None,
                          tqdm_enable: bool = True,
                          show: bool = False):
    prices_array = read_array_local(prices_filename)
    img_list = read_pickle_local(img_list_filename)
    boxes_list = read_pickle_local(boxes_array_filename)
    price_recognizer = PriceRecognizer()
    return calc_val_metrics_iter_blur(img_list, prices_array, boxes_list,
                                      price_recognizer, algo_params,
                                      blur_params, ids, show, tqdm_enable)


def calc_val_metrics_by_photo(img_list_filename: str,
                              prices_filename: str,
                              algo_params: dict,
                              version: str = "v1",
                              save_pickle: Optional[str] = None,
                              enhance: Optional[dict] = None,
                              gpu: bool = True,
                              tqdm_enable: bool = True):
    prices_dict = read_pickle_local(prices_filename)
    img_list = read_pickle_local(img_list_filename)
    price_recognizer = PriceRecognizer(version, gpu)
    return calc_val_metrics_by_photo_iter(img_list, prices_dict, price_recognizer,
                                          algo_params, save_pickle, enhance,
                                          tqdm_enable)


def calc_val_metrics_by_photo_iter(img_dict: dict, prices_dict: dict,
                                   price_recognizer: PriceRecognizer,
                                   algo_params: Dict[str, Union[int, float]],
                                   save_pickle: Optional[str] = None,
                                   enhance: Optional[dict] = None,
                                   tqdm_enable: bool = False):
    if isinstance(enhance, dict):
        save_pickle = save_pickle + enhance['suffix']
    try:
        pred_boxes_dict = read_pickle_local(save_pickle)
    except:
        iter = tqdm(img_dict.items(),
                    total=len(img_dict), disable=not tqdm_enable, smoothing=TQDM_SMOOTH)
        pred_boxes_dict = load_data_for_metrics(enhance, iter, price_recognizer)
        write_pickle_local(save_pickle, pred_boxes_dict)
    iter = enumerate(img_dict.items())
    iter = tqdm(iter, total=len(img_dict), disable=not tqdm_enable, smoothing=TQDM_SMOOTH)
    soft_metrics_total, hard_metrics_total, \
    hard_det_metrics_total, soft_det_metrics_total = [], [], [], []
    soft_metrics_micro, hard_metrics_micro, \
    hard_det_metrics_micro, soft_det_metrics_micro = [], [], [], []
    for i, (photo_id, img_list) in iter:
        iter2 = zip(img_list, pred_boxes_dict[photo_id], prices_dict[photo_id])
        soft_metrics, hard_metrics, \
        hard_det_metrics, soft_det_metrics = [], [], [], []
        for img, pred_boxes, price_true in iter2:
            price_pred, price_boxes, \
            rub_bboxes, \
            kop_bboxes = price_recognizer.recognize_price_from_boxes(img, pred_boxes,
                                                                     False, algo_params)
            price_true = round(float(price_true), 2)
            price_pred = round(float(price_pred), 2)
            soft_metric = int(price_pred) == int(price_true)
            hard_metric = price_true == price_pred
            if len(pred_boxes):
                digits_pred = [str(x) for x in pred_boxes[:, 0]]
                num_str = str(price_true).replace('.', '')
                good_detection_flag = all(dt in digits_pred for dt in num_str)
            else:
                good_detection_flag = False
            soft_metrics.append(soft_metric)
            soft_metrics_micro.append(soft_metric)
            hard_metrics.append(hard_metric)
            hard_metrics_micro.append(hard_metric)
            if good_detection_flag:
                hard_det_metrics.append(hard_metric)
                hard_det_metrics_micro.append(hard_metric)
                soft_det_metrics.append(soft_metric)
                soft_det_metrics_micro.append(soft_metric)
        soft_metrics_total.append(np.mean(soft_metrics))
        hard_metrics_total.append(np.mean(hard_metrics))
        if len(soft_det_metrics):
            soft_det_metrics_total.append(np.mean(soft_det_metrics))
            hard_det_metrics_total.append(np.mean(hard_det_metrics))
    return (np.mean(soft_metrics_total), np.mean(hard_metrics_total),
            np.mean(soft_det_metrics_total), np.mean(hard_det_metrics_total)), (
               np.mean(soft_metrics_micro), np.mean(hard_metrics_micro),
               np.mean(soft_det_metrics_micro), np.mean(hard_det_metrics_micro)), \
           len(soft_det_metrics_micro)


def load_data_for_metrics(enhance, iter, price_recognizer):
    if isinstance(enhance, dict):
        if 'lamb' in enhance:
            lamb = enhance['lamb']
            cs = enhance['cs']
            pred_boxes_dict = {key: [price_recognizer.detect(enhance_fft(
                img, lamb, cs)) for img in img_list] for key, img_list in iter}
        elif 'MIN_QUAD_AREA_RATIO' in enhance:
            MIN_QUAD_AREA_RATIO = enhance['MIN_QUAD_AREA_RATIO']
            MAX_QUAD_ANGLE_RANGE = enhance['MAX_QUAD_ANGLE_RANGE']
            docscanner = DocScanner(MIN_QUAD_AREA_RATIO=MIN_QUAD_AREA_RATIO,
                                    MAX_QUAD_ANGLE_RANGE=MAX_QUAD_ANGLE_RANGE)

            def detect(img):
                img_scan = docscanner.scan(img)
                img_scan = np.expand_dims(img_scan, -1)
                pred = price_recognizer.detect(img_scan, convert_to_gray=False)
                return pred

            pred_boxes_dict = {key: [detect(img) for img in img_list] for
                               key, img_list in iter}
        else:
            sample_fraction = enhance['sample_fraction']
            value_threshold = enhance['value_threshold']
            sat_threshold = enhance['sat_threshold']
            num_colors = enhance['num_colors']

            def detect(img):
                img_enh = enhance_noteshrink(img, sample_fraction, value_threshold,
                                             sat_threshold, num_colors)
                img_enh = np.ones_like(img_enh) * 255 - img_enh
                # show_img(img_enh)
                pred = price_recognizer.detect(np.expand_dims(img_enh, -1),
                                               convert_to_gray=False)
                return pred

            pred_boxes_dict = {key: [detect(img)
                                     for img in img_list] for key, img_list in iter}
    else:
        pred_boxes_dict = {key: [price_recognizer.detect(img) for img in img_list] for \
                           key, img_list in iter}
    return pred_boxes_dict


def load_data_for_metrics_list(enhance, iter, price_recognizer):
    if isinstance(enhance, dict):
        if 'lamb' in enhance:
            lamb = enhance['lamb']
            cs = enhance['cs']
            pred_boxes_list = [price_recognizer.detect(enhance_fft(img, lamb, cs)) for img
                               in iter]
        elif 'MIN_QUAD_AREA_RATIO' in enhance:
            MIN_QUAD_AREA_RATIO = enhance['MIN_QUAD_AREA_RATIO']
            MAX_QUAD_ANGLE_RANGE = enhance['MAX_QUAD_ANGLE_RANGE']
            docscanner = DocScanner(MIN_QUAD_AREA_RATIO=MIN_QUAD_AREA_RATIO,
                                    MAX_QUAD_ANGLE_RANGE=MAX_QUAD_ANGLE_RANGE)

            def detect(img):
                img_scan = docscanner.scan(img)
                img_scan = np.expand_dims(img_scan, -1)
                pred = price_recognizer.detect(img_scan, convert_to_gray=False)
                return pred

            pred_boxes_list = [detect(img) for img in iter]
        else:
            sample_fraction = enhance['sample_fraction']
            value_threshold = enhance['value_threshold']
            sat_threshold = enhance['sat_threshold']
            num_colors = enhance['num_colors']

            def detect(img):
                img_enh = enhance_noteshrink(img, sample_fraction, value_threshold,
                                             sat_threshold, num_colors)
                img_enh = np.ones_like(img_enh) * 255 - img_enh
                # show_img(img_enh)
                pred = price_recognizer.detect(np.expand_dims(img_enh, -1),
                                               convert_to_gray=False)
                return pred

            pred_boxes_list = [detect(img) for img in iter]
    else:
        pred_boxes_list = [price_recognizer.detect(img) for img in iter]
    return pred_boxes_list


def calc_val_metrics_iter_blur(img_list: List[np.ndarray], prices_array: np.ndarray,
                               boxes_list: List[np.ndarray],
                               price_recognizer: PriceRecognizer,
                               algo_params: Dict[str, Union[int, float]],
                               blur_params: Optional[Dict[str, float]] = None,
                               ids: Optional[List[int]] = None,
                               show: bool = False, tqdm_enable: bool = False):
    soft_metrics, hard_metrics, \
    hard_det_metrics, soft_det_metrics = [], [], [], []
    res_ids = []
    for i, (img, price_true) in tqdm(enumerate(zip(img_list, prices_array)),
                                     total=len(img_list),
                                     disable=not tqdm_enable, smoothing=TQDM_SMOOTH):
        if ids is not None and i not in ids:
            continue
        # try:
        #     rub, kop = price_recognizer.recognize(img, show)
        # except ValueError:
        #     rub, kop = [], []

        price_pred, \
        pred_boxes, price_boxes, \
        rub_digits, kop_digits = price_recognizer.recognize_float(img, algo_params)
        soft_metric = int(price_pred) == int(price_true)
        hard_metric = price_true == price_pred
        if len(pred_boxes):
            digits_pred = [str(x) for x in pred_boxes[:, 0]]
            num_str = str(float(price_true)).replace('.', '')
            good_detection_flag = all(dt in digits_pred for dt in num_str)
            # print(digits_pred, str(price_true), soft_det_metric)
        else:
            good_detection_flag = False
            hard_det_metric = True
            soft_det_metric = True
        no_kop_flag = (price_true - int(price_true)) == 0
        # if show and not soft_det_metric and no_kop_flag:
        # if show:
        # if not hard_det_metric and no_kop_flag:
        # if not hard_det_metric and not no_kop_flag:
        if not (hard_metric and good_detection_flag) and show:
            print(price_pred, price_true, soft_metric, hard_metric)
            show_boxes_on_image(pred_boxes, img)
        # if not hard_det_metric:
        #     res_ids.append(i)
        #     if show:
        #         print(price_pred, price_true, soft_metric, hard_metric)
        #         show_boxes_on_image(pred_boxes, img)
        soft_metrics.append(soft_metric)
        hard_metrics.append(hard_metric)
        if good_detection_flag:
            hard_det_metrics.append(hard_metric)
            soft_det_metrics.append(soft_metric)
        # if show:
        # if show and not soft_det_metric and no_kop_flag:
        # if show and not hard_det_metric and not no_kop_flag:
        # if not hard_det_metric and no_kop_flag:
        # print("price digits", price_boxes[:, 0])
        # print("pred digits", pred_boxes[:, 0],
        #       "rub", rub_digits, "kop", kop_digits)
        # print("Soft metrics:", np.mean(soft_metrics))
        # print("Hard metrics:", np.mean(hard_metrics))
        # print("Soft detection metrics:", np.mean(soft_det_metrics))
        # print("Hard detection metrics:", np.mean(hard_det_metrics))

    if show: print("Res ids:", res_ids)
    return np.mean(soft_metrics), np.mean(hard_metrics), \
           np.mean(soft_det_metrics), np.mean(hard_det_metrics), \
           len(soft_det_metrics), len(soft_metrics)


def run_find_price(img_list_filename: str,
                   boxes_arrays_filename: str, n: Optional[int],
                   tqdm_enable: bool = True,
                   show: bool = False, random_sample: bool = False):
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    img_list = read_pickle_local(img_list_filename)
    if n is not None:
        pairs = list(zip(boxes_array_list, img_list))
        pairs_sample = random.sample(pairs, n) if random_sample else pairs[:n]
        boxes_array_list, img_list = list(zip(*pairs_sample))
    run_find_price_iter(img_list, boxes_array_list, show, tqdm_enable)


def run_find_price_iter(img_list: List[np.ndarray], boxes_array_list: List[np.ndarray],
                        show: bool,
                        tqdm_enable: bool):
    pred_boxes_lens = []
    # ids = [1, 2, 131, 158]
    ids = None
    price_recognizer = PriceRecognizer()
    for i, (boxes_array, img) in tqdm(enumerate(zip(boxes_array_list, img_list)),
                                      total=len(img_list),
                                      disable=not tqdm_enable, smoothing=TQDM_SMOOTH):
        if ids is not None and i not in ids:
            continue
        rub, kop = price_recognizer.recognize_float(img, show=show)
        if show: print(rub, kop)

    print("num bboxes stats, when bboxes changed", np.round(np.mean(pred_boxes_lens), 5),
          np.round(np.median(pred_boxes_lens), 5), np.round(np.std(pred_boxes_lens), 5))


def run_find_price_iter2(img_list: List[np.ndarray], boxes_array_list: List[np.ndarray],
                         digits_model,
                         mser_params: Dict[str, Union[int, float]],
                         area_filter_params: Dict[str, Union[int, float]],
                         aspect_ratio_filter_params: Dict[str, float],
                         threshold_dict: Dict[str, float], max_side: int,
                         divide_algo_param_dict: Dict[str, float],
                         show: bool,
                         tqdm_enable: bool):
    # thr_angle = 35
    thr_angle = 45
    thr_distance_factor = 2
    pred_boxes_lens = []
    # ids = [1, 2, 131, 158]
    ids = None
    for i, (boxes_array, img) in tqdm(enumerate(zip(boxes_array_list, img_list)),
                                      total=len(img_list),
                                      disable=not tqdm_enable, smoothing=TQDM_SMOOTH):
        if i not in ids:
            continue
        pred_boxes = detect_digits(img, digits_model, mser_params, area_filter_params,
                                   aspect_ratio_filter_params, threshold_dict, 1, None,
                                   max_side, False,
                                   divide_algo_param_dict=divide_algo_param_dict)
        if len(pred_boxes):
            h, w = img.shape[:2]
            price_boxes = find_price(pred_boxes, h, w, thr_angle,
                                     thr_distance_factor, False)
            if np.all(price_boxes != pred_boxes):
                pred_boxes_lens.append(len(pred_boxes))
                if show:
                    find_price(pred_boxes, h, w, thr_angle,
                               thr_distance_factor, True)
                    show_boxes_on_image(pred_boxes, img)
                    show_boxes_on_image(price_boxes, img)
    print("num bboxes stats, when bboxes changed", np.round(np.mean(pred_boxes_lens), 5),
          np.round(np.median(pred_boxes_lens), 5), np.round(np.std(pred_boxes_lens), 5))


def calc_val_metrics_many_prices(img_list_filename: str,
                                 prices_filename: str,
                                 algo_params: dict,
                                 version: str = "v1",
                                 enhance: Optional[dict] = None,
                                 ids: Optional[list] = None,
                                 save_pickle: Optional[str] = None,
                                 show: bool = False,
                                 gpu: bool = True,
                                 tqdm_enable: bool = True):
    prices_dict = read_array_local(prices_filename)
    img_list = read_array_local(img_list_filename)
    price_recognizer = PriceRecognizer(version, gpu)
    return calc_val_metrics_many_prices_iter(img_list, prices_dict, price_recognizer,
                                             algo_params, enhance, ids, save_pickle,
                                             show, tqdm_enable)


def calc_val_metrics_many_prices_iter(img_list: np.ndarray, prices_array: np.ndarray,
                                      price_recognizer: PriceRecognizer,
                                      algo_params: Dict[str, Union[int, float]],
                                      enhance: Optional[dict] = None,
                                      ids: Optional[List[int]] = None,
                                      save_pickle: Optional[str] = None,
                                      show: bool = False, tqdm_enable: bool = False):
    if isinstance(enhance, dict):
        save_pickle = save_pickle + enhance['suffix']
    try:
        pred_boxes_list = read_pickle_local(save_pickle)
    except:
        iter = tqdm(img_list,
                    total=len(img_list), disable=not tqdm_enable, smoothing=TQDM_SMOOTH)
        pred_boxes_list = load_data_for_metrics_list(enhance, iter, price_recognizer)
        write_pickle_local(save_pickle, pred_boxes_list)
    iter = enumerate(zip(img_list, prices_array, pred_boxes_list))
    soft_metrics, hard_metrics, \
    hard_det_metrics, soft_det_metrics, detections = [], [], [], [], []
    res_ids = []
    for i, (img, price_list_true, pred_boxes) in tqdm(iter, total=len(img_list),
                                                      disable=not tqdm_enable,
                                                      smoothing=TQDM_SMOOTH):
        if ids is not None and i not in ids:
            continue
        price_pred, \
        rub_bboxes, \
        kop_bboxes = price_recognizer.recognize_price_from_boxes(img, pred_boxes,
                                                                 False, algo_params)
        if len(price_list_true) > 1 and len(rub_bboxes):
            img_h, img_w = img.shape[:2]
            rub_bboxes2 = rub_bboxes.copy()
            if kop_bboxes is not None:
                kop_bboxes2 = kop_bboxes.copy()
                digit_boxes_cur = np.concatenate((rub_bboxes2, kop_bboxes2))
            else:
                digit_boxes_cur = rub_bboxes2
            left_top_points = np.array(digit_boxes_cur[:, (1, 2)])
            center_x = left_top_points[:, 0].mean() / img_w
            center_y = left_top_points[:, 1].mean() / img_h
            price_dists = [dist(center_x, center_y, x, y) for (x, y), _ in
                           price_list_true]
            price_true = price_list_true[np.argmin(price_dists)][1]
        else:
            if len(price_list_true):
                price_true = price_list_true[0][1]
            else:
                continue

        price_true = round(float(price_true), 2)
        if price_true == 0.:
            continue
        price_pred = round(float(price_pred), 2)
        soft_metric = int(price_pred) == int(price_true)
        if soft_metric:
            if (price_pred - int(price_pred)) > 0. and (price_true - int(price_pred)
                                                        == 0.):
                hard_metric = True
            else:
                hard_metric = price_true == price_pred
        else:
            hard_metric = price_true == price_pred
        if len(pred_boxes):
            digits_pred = [str(x) for x in pred_boxes[:, 0]]
            num_str = str(price_true).replace('.', '')
            digits_counter = Counter(digits_pred)
            digits_counter_true = Counter(num_str)
            good_detection_flag = all(digits_counter[k] >= v for k, v
                                      in digits_counter_true.items())
        else:
            good_detection_flag = False
        # show_flag = not hard_metric and good_detection_flag and len(str(int(
        #     price_pred))) == 1
        show_flag = not hard_metric
        if show and show_flag:
            price_recognizer.recognize_price_from_boxes(img, pred_boxes,
                                                        True, algo_params)
            print("price true:", price_true)
            print("price pred:", price_pred,
                  "Rub metrics:", soft_metric, "Rub+kop metric:", hard_metric)
            print("pred boxes:", pred_boxes)
            resize_boxes(img, pred_boxes, price_recognizer.max_side)
            show_boxes_on_image(pred_boxes, img)
        soft_metrics.append(soft_metric)
        hard_metrics.append(hard_metric)
        if good_detection_flag:
            hard_det_metrics.append(hard_metric)
            soft_det_metrics.append(soft_metric)

    if show: print("Res ids:", res_ids)
    return np.mean(soft_metrics), np.mean(hard_metrics), \
           np.mean(soft_det_metrics), np.mean(hard_det_metrics), \
           len(soft_det_metrics)


def calc_val_metrics_many_prices2(img_list_filename: str,
                                  prices_filename: str,
                                  trees_filename: str,
                                  algo_params: dict,
                                  version: str = "v1",
                                  enhance: Optional[dict] = None,
                                  save_pickle: Optional[str] = None,
                                  show: bool = False,
                                  gpu: bool = True,
                                  tqdm_enable: bool = True,
                                  n: Optional[int] = None):
    prices_array = read_array_local(prices_filename)
    img_list = read_array_local(img_list_filename)
    if n is not None:
        img_list = img_list[:n]
    trees_dict = read_pickle_local(trees_filename)
    price_recognizer = PriceRecognizer(version, gpu, trees_dict=trees_dict)
    return calc_val_metrics_many_prices_iter2(img_list, prices_array, price_recognizer,
                                              algo_params, enhance,
                                              None, save_pickle, show, tqdm_enable)


def calc_val_metrics_many_prices_iter2(img_list: np.ndarray, prices_array: np.ndarray,
                                       price_recognizer: PriceRecognizer,
                                       algo_params: Dict[str, Union[int, float]],
                                       enhance: Optional[dict] = None,
                                       ids: Optional[List[int]] = None,
                                       save_pickle: Optional[str] = None,
                                       show: bool = False, tqdm_enable: bool = False):
    if isinstance(enhance, dict):
        save_pickle = save_pickle + enhance['suffix']
    try:
        pred_boxes_list = read_pickle_local(save_pickle)
    except:
        iter = tqdm(img_list,
                    total=len(img_list), disable=not tqdm_enable, smoothing=TQDM_SMOOTH)
        pred_boxes_list = load_data_for_metrics_list(enhance, iter, price_recognizer)
        write_pickle_local(save_pickle, pred_boxes_list)
    iter = enumerate(zip(img_list, prices_array, pred_boxes_list))
    res_ids = []
    soft_metric_micro, soft_metric_macro = 0, []
    hard_metric_micro, hard_metric_macro = 0, []
    soft_det_metrics_micro, soft_det_metrics_macro = 0, []
    hard_det_metrics_micro, hard_det_metrics_macro = 0, []
    num_tags, num_det_tags = 0, 0
    zero_rub_count = 0
    good_count = 0
    print("len(img_list)", len(img_list))
    bad_count = 0
    for i, (img, price_list_true, pred_boxes) in tqdm(iter, total=len(img_list),
                                                      disable=not tqdm_enable,
                                                      smoothing=TQDM_SMOOTH):
        if ids is not None and i not in ids:
            continue
        tag_list = price_recognizer.recognize_price_from_boxes_v2(img, pred_boxes, False,
                                                                  algo_params)
        if not len(price_list_true):
            continue
        soft_metric_macro_list = []
        hard_metric_macro_list = []
        hard_det_metrics_macro_list = []
        soft_det_metrics_macro_list = []
        # if not tag_list:
        #     bad_count += 1
        # if show and not tag_list:
        #     print("price true:", price_list_true)
        #     price_recognizer.recognize_price_from_boxes_v2(img, pred_boxes,
        #                                                    True, algo_params)
        #     pred_boxes2 = pred_boxes.copy()
        #     resize_boxes(img, pred_boxes2, price_recognizer.max_side)
        #     show_boxes_on_image(pred_boxes2, img)
        #     print("End")

        # bad_flag = (len(price_list_true) - len(
        #     tag_list) > 0) and len(price_list_true) > 1 and len(tag_list) > 0
        # if bad_flag:
        #     bad_count += len(price_list_true) - len(tag_list)
        #     pred_boxes2 = pred_boxes.copy()
        #     resize_boxes(img, pred_boxes2, price_recognizer.max_side)
        #     show_boxes_on_image(pred_boxes2, img)
        #     print("End")

        good_det_flags = []
        for price_pred, rub_bboxes, kop_bboxes in tag_list:
            if not len(rub_bboxes):
                continue
            if len(price_list_true) > 1:
                img_h, img_w = img.shape[:2]
                rub_bboxes2 = rub_bboxes.copy()
                kop_bboxes2 = kop_bboxes.copy()
                resize_boxes(img, rub_bboxes2, price_recognizer.max_side)
                resize_boxes(img, kop_bboxes2, price_recognizer.max_side)
                digit_boxes_cur = np.concatenate((rub_bboxes2, kop_bboxes2))
                left_top_points = np.array(digit_boxes_cur[:, (1, 2)])
                center_x = left_top_points[:, 0].mean() / img_w
                center_y = left_top_points[:, 1].mean() / img_h
                price_dists = [dist(center_x, center_y, x, y) for (x, y), _ in
                               price_list_true]
                price_true = price_list_true[np.argmin(price_dists)][1]
            else:
                price_true = price_list_true[0][1]
            price_true = round(float(price_true), 2)
            if price_true == 0.:
                continue
            price_pred = round(float(price_pred), 2)
            if len(pred_boxes):
                digits_pred = [str(x) for x in pred_boxes[:, 0]]
                num_str = str(price_true).replace('.', '')
                digits_counter = Counter(digits_pred)
                digits_counter_true = Counter(num_str)
                good_detection_flag = all(digits_counter[k] >= v for k, v
                                          in digits_counter_true.items())
            else:
                good_detection_flag = False
            # price_pred_list = [x for x in str(price_pred) if x != '.']
            # if sum(x == '1' for x in str(price_pred_list)) == len(price_pred_list):
            #     continue

            soft_metric = int(price_pred) == int(price_true)
            soft_metric_micro += soft_metric
            soft_metric_macro_list.append(soft_metric)
            if soft_metric:
                if (price_pred - int(price_pred)) > 0. and (price_true - int(price_pred)
                                                            == 0.):
                    hard_metric = True
                else:
                    hard_metric = price_true == price_pred
            else:
                hard_metric = price_true == price_pred
            # if not soft_metric and len(price_list_true) == 1:
            #     bad_count += 1
            hard_metric_micro += hard_metric
            hard_metric_macro_list.append(hard_metric)
            good_det_flags.append(good_detection_flag)
            # if not soft_metric and len(price_list_true) >= 2:
            # if show and not hard_metric:
            # bad_flag = not soft_metric and not good_detection_flag and len(
            #     tag_list) == 1 and len(price_list_true) == 1
            # bad_flag = not soft_metric and \
            #            len(tag_list) >= 1 and len(price_list_true) == 1
            bad_flag = not soft_metric and len(price_list_true) >= 1 and \
                       good_detection_flag
            # bad_flag = not hard_metric
            # if bad_flag:
            #     bad_count += 1
            if show and bad_flag:
                bad_count += 1
                price_recognizer.recognize_price_from_boxes_v2(img, pred_boxes,
                                                               True, algo_params)
                if len(price_list_true) > 1:
                    print("(center_x, center_y)", (center_x, center_y), price_list_true,
                          img.shape)
                print("price true:", price_true)
                print("rub_bboxes", rub_bboxes, "kop_bboxes", kop_bboxes)
                print("price pred:", price_pred)
                pred_boxes2 = pred_boxes.copy()
                resize_boxes(img, pred_boxes2, price_recognizer.max_side)
                show_boxes_on_image(pred_boxes2, img)
            if good_detection_flag:
                hard_det_metrics_macro_list.append(hard_metric)
                hard_det_metrics_micro += hard_metric
                soft_det_metrics_macro_list.append(soft_metric)
                soft_det_metrics_micro += soft_metric
        good_count += all(good_det_flags)
        if len(soft_metric_macro_list):
            num_tags += len(soft_metric_macro_list)
            soft_metric_macro.append(np.mean(soft_metric_macro_list))
            hard_metric_macro.append(np.mean(hard_metric_macro_list))
        else:
            # num_tags += len(price_list_true)
            num_tags += 1
            soft_metric_macro.append(0.)
            hard_metric_macro.append(0.)

        if all(good_det_flags) and len(soft_det_metrics_macro_list):
            num_det_tags += len(soft_det_metrics_macro_list)
            hard_det_metrics_macro.append(np.mean(hard_det_metrics_macro_list))
            soft_det_metrics_macro.append(np.mean(soft_det_metrics_macro_list))
    print("zero rub:", zero_rub_count, "bad", bad_count, "num tags", num_tags)
    hard_metric_micro /= num_tags
    soft_metric_micro /= num_tags
    hard_det_metrics_micro /= num_det_tags
    soft_det_metrics_micro /= num_det_tags
    if show: print("Res ids:", res_ids)
    print(len(hard_metric_macro))
    return soft_metric_micro, hard_metric_micro, \
           soft_det_metrics_micro, \
           hard_det_metrics_micro, np.mean(soft_metric_macro), np.mean(
        hard_metric_macro), np.mean(soft_det_metrics_macro), np.mean(
        hard_det_metrics_macro), good_count
