import json
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from hyperopt import Trials, fmin, tpe
from tensorflow.keras.models import load_model

from price_detector.data_processing import CLASSES_DICT, read_array_local
from price_detector.recognizer import PriceRecognizer
from price_detector.validation import calc_val_metrics_iter

MAIN_DIR = Path('..')
MODELS_DIR = MAIN_DIR / 'models'
DATASETS_DIR = MAIN_DIR / 'datasets'
PNG_FORMAT = "%s.png"

POSSIBLE_CLASSES = CLASSES_DICT.values()


def search_hyperparams_detection(model_name: str, img_list_filename: str,
                                 boxes_arrays_filename: str, space: list,
                                 search_max_evals: int, loss_threshold: float,
                                 save_suffix: str,
                                 trials_file: Optional[str] = None,
                                 gt_thresh: float = .2, max_queue_len=2,
                                 save_every_n_epoch=40,
                                 convert_to_int=False):
    digits_model = load_model(MODELS_DIR / model_name)
    boxes_array_list = read_array_local(boxes_arrays_filename)
    for i, boxes in enumerate(boxes_array_list):
        boxes_array_list[i] = np.array([b[:5] for b in boxes
                                        if len(b) == 7 and b[0] < 10])
    print(boxes_array_list.shape, boxes_array_list[0].shape)
    img_list = read_array_local(img_list_filename)

    max_side = 120

    def objective(hyperparams):
        region_box_area_ratio_threshold, \
        same_iou_threshold, iou_threshold, \
        area_ratio_threshold, same_digit_iou_threshold, \
        small_box_ratio_threshold, min_ar, max_ar, \
        min_area, max_area, \
        mser_min_area, mser_max_area, \
        mser_max_variation, mser_delta, box_expand_size = hyperparams
        if convert_to_int:
            mser_min_area = int(mser_min_area)
            min_area = int(min_area)
        mser_params = dict(_min_area=mser_min_area,
                           _max_area=mser_max_area,
                           _max_variation=mser_max_variation,
                           _delta=int(mser_delta))
        area_filter_params = dict(min_area=min_area, max_area=max_area)

        aspect_ratio_filter_params = dict(min_ar=min_ar, max_ar=max_ar)

        threshold_dict = dict(
            region_box_area_ratio_threshold=region_box_area_ratio_threshold,
            same_iou_threshold=same_iou_threshold,
            iou_threshold=iou_threshold, area_ratio_threshold=area_ratio_threshold,
            same_digit_iou_threshold=same_digit_iou_threshold,
            small_box_ratio_threshold=small_box_ratio_threshold)
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
        round_dict(mser_params)
        round_dict(area_filter_params)
        round_dict(aspect_ratio_filter_params)
        round_dict(threshold_dict)
        round_dict(divide_algo_param_dict)

        res = calc_val_metrics_iter(img_list, boxes_array_list, digits_model, gt_thresh,
                                    mser_params, area_filter_params,
                                    aspect_ratio_filter_params, threshold_dict,
                                    box_expand_size, max_side, divide_algo_param_dict,
                                    None, False)
        _, acc_macro, _, _, _, recall_macro = res
        return -acc_macro * 100
        # return -recall_macro * 100

    return search_fmin2(loss_threshold, max_queue_len, objective, search_max_evals,
                        space, trials_file, save_suffix, save_every_n_epoch)


def round_dict(dict_, ndigits=7):
    for k, v in dict_.items():
        if isinstance(v, float):
            dict_[k] = round(v, ndigits)
        elif isinstance(v, list):
            dict_[k] = [round(x, ndigits) for x in v]


def search_hyperparams_find_price(img_list_filename: str, prices_filename: str,
                                  version: str,
                                  space: list,
                                  search_max_evals: int, loss_threshold: float,
                                  save_pickle: str,
                                  save_suffix: str,
                                  trials_file: Optional[str] = None,
                                  save_every_n_epoch=40,
                                  max_queue_len=2):
    prices_array = read_array_local(prices_filename)
    img_list = read_array_local(img_list_filename)
    print("len imgs", len(img_list))
    price_recognizer = PriceRecognizer(version, gpu=False)

    from price_detector.validation.find_price_validation import \
        calc_val_metrics_many_prices_iter
    def objective(hyperparams):
        thr_angle_price, thr_angle_rub, thr_distance_factor, \
        thr_square_angle_diff, thr_dist_diff, \
        coeffs_rub_angle, coeffs_rub_area, coeffs_rub_distance, coeffs_distance_x, \
        coeffs_price_distance, coeffs_price_area = hyperparams
        algo_params = dict(
            thr_angle_price=thr_angle_price,
            thr_angle_rub=thr_angle_rub,
            thr_distance_factor=thr_distance_factor,
            thr_square_angle_diff=thr_square_angle_diff, thr_dist_diff=thr_dist_diff,
            coeffs_price_distance=coeffs_price_distance,
            coeffs_price_area=coeffs_price_area,
            coeffs_rub_angle=coeffs_rub_angle, coeffs_rub_area=coeffs_rub_area,
            coeffs_rub_distance=coeffs_rub_distance, coeffs_distance_x=coeffs_distance_x)
        round_dict(algo_params)
        _, hard_metric, _, _, _ = calc_val_metrics_many_prices_iter(img_list,
                                                                    prices_array,
                                                                    price_recognizer,
                                                                    algo_params,
                                                                    save_pickle=save_pickle)
        return -hard_metric * 100

    return search_fmin2(loss_threshold, max_queue_len, objective, search_max_evals,
                        space, trials_file, save_suffix, save_every_n_epoch)


def search_fmin2(loss_threshold, max_queue_len, objective, search_max_evals, space,
                 trials_file, suffix, save_every_n_epoch):
    if trials_file is not None:
        with open(trials_file, 'rb') as file_:
            trials = pickle.load(file_)
    else:
        trials = Trials()
    start_epochs = len(trials.trials)
    for inc_epochs in range(save_every_n_epoch, search_max_evals - start_epochs + 1,
                            save_every_n_epoch):
        max_evals = start_epochs + inc_epochs
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    loss_threshold=loss_threshold,
                    trials=trials, max_queue_len=max_queue_len)
        pickle_file = f"checkpoints/" \
                      f"trials_{suffix}_ep{max_evals}.pickle"
        with open(pickle_file, 'wb') as file_:
            pickle.dump(trials, file_)
        with open(
                f"checkpoints/best_params_{suffix}_ep{max_evals}.json", 'w') as file_:
            result = {"Best_params": best}
            json.dump(result, file_)
    return result
