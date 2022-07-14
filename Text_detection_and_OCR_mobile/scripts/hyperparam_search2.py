import sys

sys.path.append("../..")
from tools.utils import allow_growth, enableGPU

gpu_id = 0
enableGPU(gpu_id)
allow_growth()
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

from metrics.price_detector_metrics import calc_acc_f1
from price_detector.data_processing import read_pickle_local
from price_detector.detector.detect_digits import detect_digits

MAIN_DIR = Path('..')
MODELS_DIR = MAIN_DIR / 'models'


def main1():
    model_name = 'digits_epoch-74_loss-0.1696_acc-0.9559.h5'
    max_side = 120
    img_list_filename = f"schwarzkopf-price-tags-train-imgs-size-{max_side}"
    boxes_arrays_filename = f"schwarzkopf-price-tags-train-arrays-size-{max_side}"
    gt_thresh = .2

    digits_model = load_model(MODELS_DIR / model_name)
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    img_list = read_pickle_local(img_list_filename)
    max_side = 120
    best_params = {'area_ratio_threshold': 0.7721431,
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
                   'small_box_ratio_threshold': 0.0008433}

    mser_params = dict(_delta=int(best_params["mser_delta"]),
                       _max_variation=best_params["mser_max_variation"],
                       _min_area=best_params["mser_min_area"],
                       _max_area=best_params["mser_max_area"])

    area_filter_params = dict(min_area=best_params["min_area"],
                              max_area=best_params["max_area"])

    aspect_ratio_filter_params = dict(min_ar=best_params["min_ar"],
                                      max_ar=best_params["max_ar"])
    threshold_dict = best_params
    divide_algo_param_dict = dict(
        x_coord_rel_tol=.01, y_coord_rel_tol=.05,
        min_width_ratio=.25, max_width_ratio=.55,
        min_height_ratio=.8, max_height_ratio=1.05,
        left_width_threshold1=.5,
        right_width_threshold1=1.5,
        right_width_threhshold2=2.5,
        left_dist_threshold=.1,
        right_dist_threshold=1.5
    )
    box_expand_size = 2
    # space = [.0004641, 20, 30, 40, 50, 60, 70, 80]
    space = [.0004641, 10, 15, 20, 25, 30, 35, 40, 45]
    for min_area in space:
        area_filter_params.update(min_area=min_area)
        metrics, num_metrics = calc_val_metrics_iter(img_list, boxes_array_list,
                                                     digits_model,
                                                     gt_thresh, mser_params,
                                                     area_filter_params,
                                                     aspect_ratio_filter_params,
                                                     threshold_dict, box_expand_size,
                                                     max_side, divide_algo_param_dict,
                                                     None,
                                                     None,
                                                     True)
        metrics = tuple([x * 100 for x in metrics])
        print("Min area:", min_area)
        print("acc_micro: %.1f, acc_macro: %.1f, "
              "f1_micro: %.1f, f1_macro: %.1f, pr_macro: %.1f, rec_macro: %.1f" %
              metrics)
        print("before_micro: %.1f, "
              "after_micro: %.1f" % num_metrics)


def calc_val_metrics_iter(img_list: List[np.ndarray], boxes_array_list: List[np.ndarray],
                          digits_model, gt_thresh: float,
                          mser_params: Dict[str, Union[int, float]],
                          area_filter_params: Dict[str, Union[int, float]],
                          aspect_ratio_filter_params: Dict[str, float],
                          threshold_dict: Dict[str, float], box_expand_size: int,
                          max_side: int,
                          divide_algo_param_dict: Optional[Dict[str, float]],
                          params_dict: Optional[Dict[str, float]],
                          swt_params: Optional[Dict[str, float]],
                          tqdm_enable: bool):
    y_pred = []
    before_micro = 0
    after_micro = 0
    n = 0
    for i, (boxes_array, img) in tqdm(enumerate(zip(boxes_array_list, img_list)),
                                      total=len(img_list),
                                      disable=not tqdm_enable):
        res = detect_digits(img, digits_model, mser_params, area_filter_params,
                            aspect_ratio_filter_params, threshold_dict, box_expand_size,
                            params_dict, max_side, False, optimize=True,
                            divide_algo_param_dict=divide_algo_param_dict)
        pred_boxes, before_class, after_class = res
        if before_class is not None:
            n += 1
            before_micro += before_class
            after_micro += after_class
        y_pred.append(pred_boxes)
    before_micro /= n
    after_micro /= n
    res = calc_acc_f1(boxes_array_list, y_pred, gt_thresh)
    res2 = (before_micro, after_micro)
    return res, res2


if __name__ == '__main__':
    main1()
