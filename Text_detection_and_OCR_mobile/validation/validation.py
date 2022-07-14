import cProfile
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from SkynetCV import SkynetCV
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

from bounding_box_utils.bounding_box_utils import iou
from metrics.price_detector_metrics import calc_acc_f1
from price_detector.blur.blur import calc_haar_mean_metrics, calc_mean_blur_metrics
from price_detector.data_processing.utils import read_array_local, read_json, \
    read_pickle_local, \
    write_array_local, write_pickle_local
from price_detector.detector.detect_digits import detect_digits, get_algo_params
from price_detector.detector.utils import show_boxes_on_image
from price_detector.validation.find_price_validation import resize_boxes

random.seed(42)

MAIN_DIR = Path('..')
MODELS_DIR = MAIN_DIR / 'models'
DATASETS_DIR = MAIN_DIR / 'datasets'
PNG_FORMAT = "%s.png"

FilterCallable = Callable[[float], bool]

def calc_val_metrics2(param_version: str, model_name: str,
                     img_list_filename: str, boxes_arrays_filename: str,
                     n: Optional[int], gt_thresh: float = .5,
                     divide_algo: bool = False,
                     max_side: int = 120,
                     params_dict: Optional[Dict[str, float]] = None,
                     tqdm_enable: bool = True):
    #  -> Tuple[float, float, float, float]:
    digits_model = load_model(MODELS_DIR / model_name)
    boxes_array_list = read_array_local(boxes_arrays_filename)
    for i, boxes in enumerate(boxes_array_list):
        boxes_array_list[i] = np.array([b[:5] for b in boxes
                                        if len(b) == 7 and b[0] < 10])
    img_list = read_array_local(img_list_filename)
    if n is not None:
        boxes_array_list, img_list = list(
            zip(*random.sample(list(zip(boxes_array_list, img_list)), n)))
    boxes_array_list = np.array(boxes_array_list)

    area_filter_params, aspect_ratio_filter_params, \
    divide_algo_param_dict, mser_params, \
    threshold_dict, box_expand_size = get_algo_params(
        param_version, divide_algo)

    return calc_val_metrics_iter(img_list, boxes_array_list, digits_model,
                                 gt_thresh,
                                 mser_params, area_filter_params,
                                 aspect_ratio_filter_params, threshold_dict,
                                 box_expand_size, max_side, divide_algo_param_dict,
                                 params_dict, tqdm_enable)
def calc_val_metrics(param_version: str, model_name: str,
                     img_list_filename: str, boxes_arrays_filename: str,
                     n: Optional[int], gt_thresh: float = .5,
                     divide_algo: bool = False,
                     max_side: int = 120,
                     params_dict: Optional[Dict[str, float]] = None,
                     tqdm_enable: bool = True):
    #  -> Tuple[float, float, float, float]:
    digits_model = load_model(MODELS_DIR / model_name)
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    img_list = read_pickle_local(img_list_filename)
    if n is not None:
        boxes_array_list, img_list = list(
            zip(*random.sample(list(zip(boxes_array_list, img_list)), n)))

    area_filter_params, aspect_ratio_filter_params, \
    divide_algo_param_dict, mser_params, \
    threshold_dict, box_expand_size = get_algo_params(
        param_version, divide_algo)

    return calc_val_metrics_iter(img_list, boxes_array_list, digits_model, gt_thresh,
                                 mser_params, area_filter_params,
                                 aspect_ratio_filter_params, threshold_dict,
                                 box_expand_size, max_side, divide_algo_param_dict,
                                 params_dict, tqdm_enable)


def calc_val_metrics_iter(img_list: Union[List[np.ndarray], np.ndarray],
                          boxes_array_list: Union[List[np.ndarray], np.ndarray],
                          digits_model, gt_thresh: float,
                          mser_params: Dict[str, Union[int, float]],
                          area_filter_params: Dict[str, Union[int, float]],
                          aspect_ratio_filter_params: Dict[str, float],
                          threshold_dict: Dict[str, float], box_expand_size: int,
                          max_side: int,
                          divide_algo_param_dict: Optional[Dict[str, float]],
                          params_dict: Optional[Dict[str, float]], tqdm_enable: bool):
    y_pred = []
    for i, (boxes_array, img) in tqdm(enumerate(zip(boxes_array_list, img_list)),
                                      total=len(img_list),
                                      disable=not tqdm_enable, smoothing=.01):
        pred_boxes = detect_digits(img, digits_model, mser_params, area_filter_params,
                                   aspect_ratio_filter_params, threshold_dict,
                                   box_expand_size, params_dict, max_side, False,
                                   divide_algo_param_dict=divide_algo_param_dict)
        resize_boxes(img, pred_boxes, max_side)
        y_pred.append(pred_boxes)
    return calc_acc_f1(boxes_array_list, y_pred, gt_thresh)


def calc_val_metrics_debug_classifier(version: str, model_name: str,
                                      img_list_filename: str, boxes_arrays_filename: str,
                                      n: Optional[int], gt_thresh: float = .5,
                                      divide_algo: bool = False,
                                      max_side: int = 120,
                                      params_dict: Optional[Dict[str, float]] = None,
                                      tqdm_enable: bool =
                                      True):
    #  -> Tuple[float, float, float, float]:
    digits_model = load_model(MODELS_DIR / model_name)
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    img_list = read_pickle_local(img_list_filename)
    if n is not None:
        random_sample = random.sample(list(zip(boxes_array_list, img_list)), n)
        boxes_array_list, img_list = list(zip(*random_sample))
    max_side = max_side

    area_filter_params, aspect_ratio_filter_params, \
    divide_algo_param_dict, mser_params, threshold_dict, _ = \
        get_algo_params(version, divide_algo)

    box_expand_size = 2

    pickle_save = f"saved_boxes_{version}_exp_size_{box_expand_size}"

    return calc_val_metrics_debug_classifier_iter(img_list, boxes_array_list,
                                                  digits_model, gt_thresh,
                                                  mser_params,
                                                  area_filter_params,
                                                  aspect_ratio_filter_params,
                                                  threshold_dict, box_expand_size,
                                                  max_side, divide_algo_param_dict,
                                                  params_dict, pickle_save,
                                                  tqdm_enable)


def match_iou(y_true: np.ndarray, y_pred: np.ndarray,
              gt_thresh: float) -> Tuple[int, int, int]:
    true_mask = np.ones((y_true.shape[0]), dtype=bool)
    pred_mask = np.ones((y_pred.shape[0]), dtype=bool)
    tp, fp, fn = 0, 0, 0
    if len(y_true) and len(y_pred):
        similarities = iou(y_true, y_pred, coords='corners')
        while True:
            max_iou_index = np.unravel_index(np.argmax(similarities), similarities.shape)
            max_iou = similarities[max_iou_index]
            if max_iou > gt_thresh:
                ind_x, ind_y = max_iou_index
                similarities[ind_x, :] = -1
                similarities[:, ind_y] = -1
                tp += 1
                true_mask[ind_x] = False
                pred_mask[ind_y] = False
            else:
                break
        fp += np.count_nonzero(pred_mask)
        fn += np.count_nonzero(true_mask)
    elif len(y_true) and not len(y_pred):
        fn = len(y_true)
    elif len(y_pred) and not len(y_true):
        fp = len(y_pred)
    return tp, fp, fn


def calc_val_metrics_debug_classifier_iter(img_list: List[np.ndarray],
                                           boxes_array_list: List[np.ndarray],
                                           digits_model, gt_thresh: float,
                                           mser_params: Dict[str, Union[int, float]],
                                           area_filter_params:
                                           Dict[str, Union[int, float]],
                                           aspect_ratio_filter_params: Dict[str, float],
                                           threshold_dict: Dict[str, float],
                                           box_expand_size: int, max_side: int,
                                           divide_algo_param_dict:
                                           Optional[Dict[str, float]],
                                           params_dict: Optional[Dict[str, float]],
                                           pickle_save: str,
                                           tqdm_enable: bool):
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    f1_list = []
    pr_list, rec_list = [], []
    acc_list = []

    try:
        res_pickle = read_pickle_local(pickle_save)
    except:
        res_pickle = []

        for i, img in tqdm(enumerate(img_list),
                           total=len(img_list),
                           disable=not tqdm_enable, smoothing=.01):
            res = detect_digits(img, digits_model, mser_params, area_filter_params,
                                aspect_ratio_filter_params, threshold_dict,
                                box_expand_size, params_dict, max_side, False,
                                optimize=True,
                                divide_algo_param_dict=divide_algo_param_dict)

            if len(res) < 2:
                continue
            b1, b2 = res
            res_pickle.append((i, b1, b2))
        write_pickle_local(pickle_save, res_pickle)

    for i, boxes_before_classifier, pred_boxes in res_pickle:
        # img = img_list[i]
        boxes_array = boxes_array_list[i]
        tp, fp, fn = match_iou(boxes_array[:, 1:], boxes_before_classifier, gt_thresh)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

        tp_fp_sum = tp + fp
        tp_fn_sum = tp + fn
        tp_fp_fn_sum = tp_fp_sum + fn
        acc = tp / tp_fp_fn_sum if tp_fp_fn_sum else 1
        p = tp / tp_fp_sum if tp_fp_sum else 1
        r = tp / tp_fn_sum if tp_fn_sum else 1
        p_r_sum = p + r
        f1 = 2 * p * r / p_r_sum if p_r_sum else int(tp != 0)

        # if r < 1.0:
        #     print("recall", round(r, 2))
        #     show_boxes_on_image(boxes_before_classifier, img)
        #     show_boxes_on_image(pred_boxes, img)
        #     show_boxes_on_image(boxes_array, img)

        f1_list.append(f1)
        acc_list.append(acc)
        pr_list.append(p)
        rec_list.append(r)

    # calc micro
    tp_fp_sum = tp_sum + fp_sum
    p_micro = tp_sum / tp_fp_sum if tp_fp_sum else 1
    tp_fn_sum = tp_sum + fn_sum
    r_micro = tp_sum / tp_fn_sum if tp_fn_sum else 1
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if p_micro + r_micro else int(
        tp_sum != 0)
    tp_fp_fn_sum = tp_fp_sum + fn_sum
    acc_micro = tp_sum / tp_fp_fn_sum if tp_fp_fn_sum else 1
    # calc macro
    acc_macro = float(np.mean(acc_list))
    f1_macro = float(np.mean(f1_list))
    precision_macro = float(np.mean(pr_list))
    recall_macro = float(np.mean(rec_list))
    return acc_micro, acc_macro, f1_micro, f1_macro, precision_macro, recall_macro


def calc_val_metrics_debug(version: str, model_name: str,
                           img_list_filename: str, boxes_arrays_filename: str,
                           n: Optional[int], gt_thresh: float = .5,
                           divide_algo: bool = False, max_side: int = 120,
                           img_save_dir: Optional[str] = None,
                           max_images_show: Optional[int] = None,
                           acc_filter: Optional[FilterCallable] = None,
                           ids: Optional[list] = None,
                           accuracies_save_name: Optional[str] = None,
                           recalls_save_name: Optional[str] = None,
                           params_dict: Optional[Dict[str, float]] = None,
                           blur_params: Optional[Dict[str, float]] = None,
                           tqdm_enable: bool = True):
    #  -> Tuple[float, float, float, float]:
    digits_model = load_model(MODELS_DIR / model_name)
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    img_list = read_pickle_local(img_list_filename)
    if n is not None:
        boxes_array_list, img_list = list(
            zip(*random.sample(list(zip(boxes_array_list, img_list)), n)))
    max_side = max_side

    area_filter_params, aspect_ratio_filter_params, \
    divide_algo_param_dict, mser_params, \
    threshold_dict, box_expand_size = get_algo_params(
        version, divide_algo)

    calc_val_metrics_debug_iter(img_list, boxes_array_list, digits_model, gt_thresh,
                                mser_params,
                                area_filter_params, aspect_ratio_filter_params,
                                threshold_dict, box_expand_size,
                                max_side, divide_algo_param_dict, params_dict,
                                tqdm_enable, acc_filter,
                                img_save_dir, max_images_show, ids, accuracies_save_name,
                                recalls_save_name, blur_params)


def calc_val_metrics_debug_iter(img_list: List[np.ndarray],
                                boxes_array_list: List[np.ndarray],
                                digits_model, gt_thresh: float,
                                mser_params: Dict[str, Union[int, float]],
                                area_filter_params: Dict[str, Union[int, float]],
                                aspect_ratio_filter_params: Dict[str, float],
                                threshold_dict: Dict[str, float],
                                box_expand_size: int, max_side: int,
                                divide_algo_param_dict: Optional[Dict[str, float]],
                                params_dict: Optional[Dict[str, float]],
                                tqdm_enable: bool,
                                acc_filter: Optional[FilterCallable] = None,
                                img_save_dir: Optional[str] = None,
                                max_images_show: int = None, ids: Optional[list] = None,
                                accuracies_save_name: Optional[str] = None,
                                recalls_save_name: Optional[str] = None,
                                blur_params: Optional[Dict[str, float]] = None):
    if img_save_dir is not None:
        img_save_dir = DATASETS_DIR / img_save_dir
        img_save_dir.mkdir(exist_ok=True)

    if ids is None:
        ids = range(len(img_list))
    y_pred = []
    images_showed = 0
    accuracies, recalls = [], []
    profile = cProfile.Profile()
    for i, (boxes_array, img) in tqdm(enumerate(zip(boxes_array_list, img_list)),
                                      total=len(img_list),
                                      disable=not tqdm_enable, smoothing=.01):
        if blur_params is not None:
            blur_metric = calc_mean_blur_metrics(boxes_array, blur_params["dsize"], img)
            if blur_metric[1] < blur_params["threshold"]:
                continue
        if i not in ids:
            continue
        profile.enable()
        pred_boxes = detect_digits(img, digits_model, mser_params, area_filter_params,
                                   aspect_ratio_filter_params, threshold_dict,
                                   box_expand_size, params_dict, max_side, False,
                                   divide_algo_param_dict=divide_algo_param_dict)
        resize_boxes(img, pred_boxes, max_side)
        profile.disable()
        y_pred.append(pred_boxes)
        _, acc, _, _, _, rec = calc_acc_f1([boxes_array], [pred_boxes], gt_thresh, False)
        accuracies.append(acc)
        recalls.append(rec)
        if max_images_show is None or max_images_show < images_showed:
            if acc_filter is not None and acc_filter(acc):
                # print("Acc:", acc)
                # print(boxes_array, pred_boxes)
                images_showed += 1
                if img_save_dir is not None:
                    image_save_list = [img_save_path(f"{i}_mser", img_save_dir),
                                       img_save_path(f"{i}_before_classifier",
                                                     img_save_dir),
                                       img_save_path(f"{i}_pred", img_save_dir),
                                       img_save_path(f"{i}_true", img_save_dir)]
                    detect_digits(img, digits_model, mser_params, area_filter_params,
                                  aspect_ratio_filter_params, threshold_dict,
                                  box_expand_size, params_dict, max_side, True,
                                  image_save_list,
                                  divide_algo_param_dict=divide_algo_param_dict)
                    # calc_acc_f1([boxes_array], [pred_boxes], gt_thresh, False, True)
                    show_boxes_on_image(pred_boxes, img,
                                        write_to_file=image_save_list[2])
                    show_boxes_on_image(boxes_array, img,
                                        write_to_file=image_save_list[3])
                else:
                    detect_digits(img, digits_model, mser_params, area_filter_params,
                                  aspect_ratio_filter_params, threshold_dict,
                                  box_expand_size, params_dict, max_side, True,
                                  divide_algo_param_dict=divide_algo_param_dict)
                    # calc_acc_f1([boxes_array], [pred_boxes], gt_thresh, False, True)
                    show_boxes_on_image(pred_boxes, img)
                    show_boxes_on_image(boxes_array, img)

    profile.disable()
    profile.print_stats("tottime")
    write_array_local(accuracies_save_name, np.array(accuracies))
    write_array_local(recalls_save_name, np.array(recalls))
    return calc_acc_f1(boxes_array_list, y_pred, gt_thresh)


DIGIT_DEST_SHAPE = (32, 32)


def calc_classifier_metrics(model_name: str,
                            img_list_filename: str, boxes_arrays_filename: str,
                            ids_list_filename: str,
                            save_bad_ids_filename: str,
                            blur_params: Optional[Dict[str, float]] = None,
                            ids: Optional[list] = None,
                            tqdm_enable: bool = True) -> Tuple[float, float]:
    digits_model = load_model(MODELS_DIR / model_name)
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    img_list = read_pickle_local(img_list_filename)
    id_list = read_pickle_local(ids_list_filename)
    if ids is None:
        ids = range(len(img_list))
    acc_macro, acc_micro, not_blur_count, \
    num_bad_tags, num_boxes, bad_ids = calc_classifier_metrics_iter(img_list,
                                                                    boxes_array_list,
                                                                    id_list,
                                                                    digits_model,
                                                                    blur_params, ids,
                                                                    tqdm_enable)
    write_array_local(save_bad_ids_filename, np.array(bad_ids))
    acc_macro = float(np.mean(acc_macro))
    acc_micro /= num_boxes
    print("Num bad tags:", num_bad_tags)
    print("not blur count", not_blur_count / len(img_list))
    return acc_micro, acc_macro


def calc_classifier_metrics_iter(img_list, boxes_array_list, id_list, digits_model,
                                 blur_params, ids, tqdm_enable):
    acc_micro = 0
    acc_macro = []
    num_boxes = 0
    not_blur_count = 0
    num_bad_tags = 0
    num_not_blur_good_classified = 0
    num_not_blur_bad_classified = 0
    num_imgs = len(img_list)
    iter = enumerate(zip(boxes_array_list, img_list, id_list))
    bad_ids = []
    for i, (boxes_array, img, tag_id) in tqdm(iter,
                                              total=num_imgs,
                                              disable=not tqdm_enable, smoothing=.01):
        if i not in ids:
            continue
        img_gray = SkynetCV.bgr2grayscale(img)
        # if blur_params is not None:
        #     blur_metric = calc_mean_blur_metrics(boxes_array, blur_params["dsize"], img)
        #     if blur_metric[3] < blur_params["threshold"]:
        #         continue
        not_blur_count += 1
        len_boxes = len(boxes_array)
        num_boxes += len_boxes
        boxes_coords = boxes_array[:, 1:]
        boxes_pred, classes = digit_classification(boxes_coords, digits_model,
                                                   img_gray)
        classes = [0 if x == 10 else (10 if x == 0 else x) for x in classes]
        cur_metrics = np.array(
            [boxes_true[0] == class_pred and class_pred != 10
             for boxes_true, class_pred in zip(boxes_array, classes)])
        sum_metrics = np.sum(cur_metrics)
        if sum_metrics != len_boxes:
            num_bad_tags += 1
            bad_ids.append(tag_id)
            if blur_params is not None:
                # blur_metric = calc_mean_blur_metrics(boxes_array, blur_params["dsize"],
                #                                      img)
                per_blur, blur_extent = calc_haar_mean_metrics(
                    img_gray.squeeze(), blur_params["dsize"], blur_params["threshold"])
                # if blur_metric[3] > blur_params["threshold"]:
                if per_blur > blur_params["min_zero"]:
                    num_not_blur_bad_classified += 1
                    print(per_blur, blur_extent)
                    print("true")
                    show_boxes_on_image(boxes_array, img)
                    print("pred")
                    show_boxes_on_image(boxes_coords, img, classes)
        else:
            if blur_params is not None:
                # blur_metric = calc_mean_blur_metrics(boxes_array, blur_params["dsize"],
                #                                      img)
                per_blur, blur_extent = calc_haar_mean_metrics(
                    img_gray.squeeze(), blur_params["dsize"], blur_params["threshold"])
                # if blur_metric[3] > blur_params["threshold"]:
                if per_blur > blur_params["min_zero"]:
                    num_not_blur_good_classified += 1
                    print(f"per_blur: {per_blur}, blur_extent: {blur_extent}")
                    print("true")
                    show_boxes_on_image(boxes_array, img)
                    print("pred")
                    show_boxes_on_image(boxes_coords, img, classes)
        # if blur_params is not None:
        #     # blur_metric = calc_mean_blur_metrics(boxes_array, blur_params["dsize"],
        #     #                                      img)
        #     blur_metric = calc_haar_mean_metrics(img,
        #                                          boxes_array,
        #                                          blur_params["dsize"],
        #                                          blur_params[
        #                                              "intensity_threshold"])
        #     # if blur_metric[3] > blur_params["threshold"]:
        #     if blur_metric[3] > blur_params["min_zero"]:
        #         print("true")
        #         show_boxes_on_image(boxes_array, img)
        #         print("pred")
        #         show_boxes_on_image(boxes_coords, img, classes)

        acc_micro += sum_metrics
        acc_macro.append(sum_metrics / len_boxes)
        # if sum_metrics != len_boxes:
        #     print(sum_metrics / len_boxes)
        #     show_boxes_on_image(boxes_pred, img, classes)
        #     show_boxes_on_image(boxes_array, img)
    num_good_tags = num_imgs - num_bad_tags
    good_classified_not_blur_ratio = round(num_not_blur_good_classified /
                                           num_good_tags, 4)
    bad_classified_not_blur_ratio = round(num_not_blur_bad_classified / num_bad_tags, 4)
    print(f"Good classsified not blur ratio: "
          f"{good_classified_not_blur_ratio};"
          f"Bad classsified not blur ratio: {bad_classified_not_blur_ratio}")
    return acc_macro, acc_micro, not_blur_count, num_bad_tags, num_boxes, bad_ids


def digit_classification(boxes: np.ndarray,
                         digit_classifier_model,
                         img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_batch = []
    batch_shape = DIGIT_DEST_SHAPE + (1,)
    w, h = DIGIT_DEST_SHAPE
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        digit = img[ymin:ymax, xmin:xmax]
        digit = SkynetCV.resize(digit, w, h)
        digit = digit.reshape(batch_shape)
        img_batch.append(digit)
    img_batch = np.asarray(img_batch)
    preds = digit_classifier_model.predict(img_batch, batch_size=128)
    boxes, classes = get_classes_from_boxes_preds(boxes, preds)
    return boxes, classes


def get_classes_from_boxes_preds(boxes, preds):
    classes = []
    for i, pred in enumerate(preds):
        pred_class = pred.argmax()
        classes.append(pred_class)
    return boxes, np.array(classes)


def run_detector(version: str, model_name: str,
                 img_list_filename: str, img_save_dir: str,
                 max_good: int, max_bad: int,
                 divide_algo: bool = False,
                 max_side: int = 120,
                 tqdm_enable: bool = True):
    digits_model = load_model(MODELS_DIR / model_name)
    img_list = read_pickle_local(img_list_filename)
    max_side = max_side

    area_filter_params, aspect_ratio_filter_params, \
    divide_algo_param_dict, mser_params, \
    threshold_dict, box_expand_size = get_algo_params(
        version, divide_algo)

    return iter_detector_imgs(img_list, img_save_dir, digits_model, mser_params,
                              area_filter_params,
                              aspect_ratio_filter_params, threshold_dict,
                              box_expand_size, max_side,
                              divide_algo_param_dict,
                              max_good, max_bad, tqdm_enable)


def iter_detector_imgs(img_list: List[np.ndarray], img_save_dir: str, digits_model,
                       mser_params: Dict[str, Union[int, float]],
                       area_filter_params: Dict[str, Union[int, float]],
                       aspect_ratio_filter_params: Dict[str, float],
                       threshold_dict: Dict[str, float], box_expand_size, max_side: int,
                       divide_algo_param_dict: Optional[Dict[str, float]],
                       max_good: int, max_bad: int,
                       tqdm_enable: bool):
    img_save_dir = DATASETS_DIR / img_save_dir
    good_img_save_dir = img_save_dir / 'good'
    good_img_save_dir.mkdir(exist_ok=True, parents=True)
    bad_img_save_dir = img_save_dir / 'bad'
    bad_img_save_dir.mkdir(exist_ok=True)
    good_count, bad_count = -1, -1
    random.shuffle(img_list)
    for i, img in tqdm(enumerate(img_list),
                       total=len(img_list),
                       disable=not tqdm_enable, smoothing=.01):
        pred_boxes = detect_digits(img, digits_model, mser_params, area_filter_params,
                                   aspect_ratio_filter_params, threshold_dict,
                                   box_expand_size, None, max_side, False,
                                   divide_algo_param_dict=divide_algo_param_dict)
        if len(pred_boxes):
            good_count += 1
            if good_count >= max_good:
                continue
            subdir_path = good_img_save_dir
            bad_flag = False
        else:
            bad_count += 1
            if bad_count >= max_bad:
                continue
            subdir_path = bad_img_save_dir
            bad_flag = True
        image_save_list = [img_save_path(f"{i}_mser", subdir_path),
                           img_save_path(f"{i}_before_classifier",
                                         subdir_path),
                           img_save_path(f"{i}_pred", subdir_path)]
        detect_digits(img, digits_model, mser_params, area_filter_params,
                      aspect_ratio_filter_params, threshold_dict, box_expand_size, None,
                      max_side, True, image_save_list,
                      divide_algo_param_dict=divide_algo_param_dict)
        if not bad_flag:
            show_boxes_on_image(pred_boxes, img,
                                write_to_file=image_save_list[2])
    print("Bad count:", bad_count)


def img_save_path(name, img_save_dir):
    return str(img_save_dir / (PNG_FORMAT % name))


def parse_via_json_validation(annotation_file1: str,
                              annotation_file2: str):
    annotation = read_json(annotation_file1)
    annotation2 = read_json(annotation_file2)
    res_dict = defaultdict(list)
    parse_annotation_dict_validation(annotation, res_dict)
    parse_annotation_dict_validation(annotation2, res_dict)
    counter = defaultdict(int)
    rub_accs, rub_kop_accs = [], []
    for photo_id, labels in res_dict.items():
        rub_acc, rub_kop_acc, num_tags = 0, 0, 0
        for label in labels:
            counter[label] += 1
            if label == 'rub':
                rub_acc += 1
                num_tags += 1
            elif label == 'kop':
                rub_acc += 1
                rub_kop_acc += 1
                num_tags += 1
            elif label == 'error':
                num_tags += 1
        rub_accs.append(rub_acc / num_tags)
        rub_kop_accs.append(rub_kop_acc / num_tags)
    print("Rub acc:", round(np.mean(rub_accs) * 100, 3),
          "Rub+kop acc:", round(np.mean(rub_kop_accs) * 100, 3))
    print(counter)


def parse_annotation_dict_validation(annotation_dict: dict, res_dict):
    for label_dict in annotation_dict.values():
        file_attrs = label_dict["file_attributes"]
        photo_id = file_attrs["photo_id"]
        val = file_attrs["validate"]
        res_dict[photo_id].append(val)
