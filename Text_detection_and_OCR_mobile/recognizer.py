from price_detector.detector.box_utils import box_areas
from tools.utils import enableGPU

# enableGPU(3)

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from tensorflow_core.python.keras.saving.save import load_model

from price_detector.data_processing import digits_to_number
from price_detector.detector.detect_digits import DIGITS_MAP, DIGIT_DEST_SHAPE, \
    crop_for_classifier, detect_digits, get_algo_params
from price_detector.detector.utils import show_boxes_on_image
from price_detector.find_price import find_price, find_price_v2, find_rub_kop_parts

MAIN_DIR = Path('..')
MODELS_DIR = MAIN_DIR / 'models'


class PriceRecognizer:
    def __init__(self, version="v11", gpu=True, trees_dict: Optional[dict] = None):
        model_name = '/home/ml/models/schwarzkopf-retail/digits_epoch-74_loss-0.1696_acc-0.9559.h5'
        self.max_side = 120
        self.area_filter_params, self.aspect_ratio_filter_params, \
        self.divide_algo_param_dict, self.mser_params, \
        self.threshold_dict, self.box_expand_size = get_algo_params(
            version, True)
        self.params_default = {'coeffs_distance_x': 1.759178,
                               'coeffs_price_area': 1.9980584,
                               'coeffs_price_distance': 0.3724674,
                               'coeffs_rub_angle': 1.1889965,
                               'coeffs_rub_area': 0.5479822,
                               'coeffs_rub_distance': 1.5264188, 'thr_angle_price': 43,
                               'thr_angle_rub': 35, 'thr_dist_diff': 0.6616423,
                               'thr_distance_factor': 0.900928,
                               'thr_square_angle_diff': 0.6608452}

        self.params_tree = {'thr_angle_price': 43, 'thr_distance_factor': 2.35,
                            'not_price_class_threshold': .01}
        if trees_dict is not None:
            self.trees_dict = trees_dict

        if gpu:
            self.digits_model = load_model(model_name)

    def recognize_float(self, img,
                        params: Optional[Dict[str, Union[int, float,
                                                         Tuple[float, float,
                                                               float, float],
                                                         Tuple[float, float]]]] = None,
                        show: bool = False):
        pred_boxes = self.detect(img, show=show)
        return self.recognize_price_from_boxes(img, pred_boxes, params)

    def recognize_price_from_boxes_v2(self, img, pred_boxes, verbose, params=None):
        if params is None:
            params = self.params_tree

        thr_angle_price = params["thr_angle_price"]
        thr_distance_factor = params["thr_distance_factor"]
        not_price_class_threshold = params["not_price_class_threshold"]
        h, w = self.get_resized_shape(img)
        if len(pred_boxes):
            tag_tuples = find_price_v2(pred_boxes, h, w, self.trees_dict, thr_angle_price,
                                       thr_distance_factor, not_price_class_threshold,
                                       img=img if verbose else None, verbose=False)
            res = []
            for rub_bboxes, kop_bboxs in tag_tuples:
                rub_digits = rub_bboxes[:, 0]
                if kop_bboxs is not None:
                    kop_digits = list(kop_bboxs[:, 0])
                    if len(kop_digits) < 2:
                        kop_digits += [0]
                    kop_float = digits_to_number(kop_digits) / 100
                else:
                    kop_float = 0.
                rub_float = digits_to_number(rub_digits)
                price_float = round(rub_float + kop_float, 2)
                res.append((price_float, rub_bboxes, kop_bboxs))
            return res
        else:
            return [[0., [], []]]

    def recognize_price_from_boxes(self, img, pred_boxes, show,
                                   params=None, stage2=False):
        if params is None:
            params = self.params_default

        thr_angle_price = params["thr_angle_price"]
        thr_angle_rub = params["thr_angle_rub"]
        thr_distance_factor = params["thr_distance_factor"]
        thr_square_angle_diff = params["thr_square_angle_diff"]
        thr_dist_diff = params["thr_dist_diff"]
        coeffs_price_distance = params["coeffs_price_distance"]
        coeffs_price_area = params["coeffs_price_area"]
        coeffs_rub_angle = params["coeffs_rub_angle"]
        coeffs_rub_area = params["coeffs_rub_area"]
        coeffs_rub_distance = params["coeffs_rub_distance"]
        coeffs_distance_x = params["coeffs_distance_x"]
        h, w = self.get_resized_shape(img)
        if not len(pred_boxes):
            return 0., [], []
        price_boxes = find_price(pred_boxes, h, w, coeffs_price_distance,
                                 coeffs_price_area,
                                 thr_angle_price, thr_distance_factor, show)
        rub_bboxes, \
        kop_bboxs = find_rub_kop_parts(price_boxes, coeffs_rub_angle, coeffs_rub_area,
                                       coeffs_rub_distance, coeffs_distance_x,
                                       thr_angle_rub, thr_square_angle_diff,
                                       thr_dist_diff, False)
        if stage2:
            more_boxes_params_dict = {
                "kop_width_ratio": .5,
                "kop_height_ratio": .5,
            }
            if show: print("kop_bboxs1", kop_bboxs)
            kop_width_ratio = more_boxes_params_dict["kop_width_ratio"]
            kop_height_ratio = more_boxes_params_dict["kop_height_ratio"]
            rub_bboxes, kop_bboxs = self.stage2(rub_bboxes, kop_bboxs,
                                                img,
                                                kop_width_ratio,
                                                kop_height_ratio, show)
            if show:
                print("kop_bboxs2", kop_bboxs)
        rub_bboxes = self.filter_rub_boxes(rub_bboxes)
        if not len(rub_bboxes):
            return 0., [], []
        rub_digits = rub_bboxes[:, 0]
        if kop_bboxs is not None:
            kop_digits = list(kop_bboxs[:, 0])
            if len(kop_digits) < 2:
                kop_digits += [0]
            kop_float = digits_to_number(kop_digits) / 100
        else:
            kop_float = 0.
        rub_float = digits_to_number(rub_digits)
        price_float = round(rub_float + kop_float, 2)

        scale_x, scale_y = self.get_scales(img)
        resize_boxes(rub_bboxes, scale_x, scale_y)
        resize_boxes(kop_bboxs, scale_x, scale_y)
        if show:
            print("rub_bboxes, kop_bboxs", rub_bboxes, kop_bboxs)
        return price_float, rub_bboxes, kop_bboxs

    def filter_rub_boxes(self, boxes: np.ndarray, area_ratio: float = .4):
        box_areas_list = box_areas(boxes[:, 1:])
        area_threshold = area_ratio * np.mean(box_areas_list)
        return np.array([box for box_ar, box in zip(box_areas_list, boxes) if
                         box_ar > area_threshold])

    def get_scales(self, img):
        img_h, img_w, _ = img.shape
        h, w = self.get_resized_shape(img)
        return img_w / w, img_h / h

    def get_resized_shape(self, img: np.ndarray):
        img_h, img_w, _ = img.shape
        if img_h > img_w:
            h = self.max_side
            w = int(h / img_h * img_w)
        else:
            w = self.max_side
            h = int(w / img_w * img_h)
        return h, w

    def stage2(self, rub_bboxes: np.ndarray,
               kop_bboxs: np.ndarray, img: np.ndarray,
               kop_width_ratio: float, kop_height_ratio: float, show):
        def run_classifier_local(box):
            pred = self.digits_model.predict(crop_for_classifier(img, digit_h,
                                                                 digit_w,
                                                                 1., 1.,
                                                                 box, batch_shape))
            pred_class = pred[0].argmax()
            return DIGITS_MAP[pred_class] if pred_class else pred_class

        def add_class(box_, pred_class):
            return np.insert(box_, 0, pred_class, axis=0)

        rub_kop_dist = 2
        kop_x_dist = 2
        img_h, img_w, _ = img.shape
        batch_shape = (1,) + DIGIT_DEST_SHAPE + (1,)
        digit_w, digit_h = DIGIT_DEST_SHAPE

        rub_bboxes = rub_bboxes[rub_bboxes[:, 1].argsort()]
        right_rub = rub_bboxes[-1]
        right_rub_xmax = right_rub[3]
        scale_x, scale_y = self.get_scales(img)
        resize_boxes(rub_bboxes, scale_x, scale_y)
        resize_boxes(kop_bboxs, scale_x, scale_y)
        if kop_bboxs is None:
            rub_width = (rub_bboxes[:, 3] - rub_bboxes[:, 1]).mean()
            kop_width = int(kop_width_ratio * rub_width)
            rub_height = (rub_bboxes[:, 4] - rub_bboxes[:, 2]).mean()
            kop_height = int(kop_height_ratio * rub_height)
            if right_rub_xmax + 2 * kop_width + rub_kop_dist + kop_x_dist < img_w:
                kop_xmin2 = right_rub_xmax + rub_kop_dist
                kop_ymin2 = right_rub[2]
                kop_xmax2 = kop_xmin2 + kop_width
                kop_ymax2 = kop_ymin2 + kop_height
                box = np.array([kop_xmin2, kop_ymin2, kop_xmax2, kop_ymax2])
                if show: show_boxes_on_image(np.array([box]), img.squeeze())
                pred_class = run_classifier_local(box)
                if show: print("box", box, "class", pred_class)
                if pred_class != 0:
                    box = add_class(box, pred_class)
                    kop_bboxs = [box]
                    kop_xmin2 = kop_xmax2 + kop_x_dist
                    box2 = np.array([kop_xmin2, kop_ymin2,
                                     kop_xmin2 + kop_width,
                                     kop_ymax2])
                    pred_class = run_classifier_local(box2)
                    if pred_class != 0:
                        box2 = add_class(box2, pred_class)
                        kop_bboxs.append(box2)
                        if show:
                            print("box", box2, "class", pred_class)
                            print("kop_bboxs2_1", kop_bboxs)
                    kop_bboxs = np.array(kop_bboxs)
                    if len(kop_bboxs) < 2:
                        right_rub_ymax = right_rub[4]
                        kop_ymin2 = right_rub_ymax - kop_height
                        box = np.array([kop_xmin2, kop_ymin2, kop_xmax2, right_rub_ymax])
                        pred_class = run_classifier_local(box)
                        kop_bboxs2 = []
                        if pred_class != 0:
                            box = add_class(box, pred_class)
                            kop_bboxs2.append(box)
                            kop_xmin2 = kop_xmax2 + kop_x_dist
                            box2 = np.array([kop_xmin2, kop_ymin2,
                                             kop_xmin2 + kop_width,
                                             kop_ymax2])
                            pred_class = run_classifier_local(box2)
                            if pred_class != 0:
                                box2 = add_class(box2, pred_class)
                                kop_bboxs2.append(box2)
                        if (kop_bboxs2 and not kop_bboxs) or len(kop_bboxs2) == 2:
                            kop_bboxs = np.array(kop_bboxs2)
        elif len(kop_bboxs) == 1:
            kop_box = kop_bboxs[0]
            _, kop_xmin1, kop_ymin1, kop_xmax1, kop_ymax1 = kop_box
            kop_width = kop_xmax1 - kop_xmin1
            if right_rub_xmax + rub_kop_dist + kop_x_dist + kop_width < kop_xmin1:
                kop_xmin2 = kop_xmin1 - kop_x_dist - kop_width
                kop_xmax2 = kop_xmin2 + kop_width
                box = np.array([kop_xmin2, kop_ymin1, kop_xmax2, kop_ymax1])
                pred_class = run_classifier_local(box)
                if show: show_boxes_on_image(np.array([box]), img.squeeze())
                if show: print("box", box, "class", pred_class)
                if pred_class != 0:
                    box = add_class(box, pred_class)
                    kop_bboxs = np.array([box, kop_bboxs[0]])
            elif kop_xmax1 + kop_x_dist + kop_width < img_w:
                kop_xmin2 = kop_xmax1 + kop_x_dist
                kop_xmax2 = kop_xmin2 + kop_width
                box = np.array([kop_xmin2, kop_ymin1, kop_xmax2, kop_ymax1])
                pred_class = run_classifier_local(box)
                if show: show_boxes_on_image(np.array([box]), img.squeeze())
                if show: print("box", box, "class", pred_class)
                if pred_class != 0:
                    box = add_class(box, pred_class)
                    kop_bboxs = np.array([kop_bboxs[0], box])
        # kop_bboxs = kop_bboxs[kop_bboxs[:, 0].argsort()]
        # left_kop_xmin = kop_bboxs[0]
        # rub_width = 1
        # if right_rub_xmax + rub_kop_dist + rub_width < left_kop_xmin:
        #     pass
        # elif left_rub_xmin - rub_width - rub_dist > 0:
        #     pass
        return rub_bboxes, kop_bboxs

    def detect(self, img, return_image=False, show=False,
               convert_to_gray=True, check_validity=True):
        return detect_digits(img, self.digits_model, self.mser_params,
                             self.area_filter_params, self.aspect_ratio_filter_params,
                             self.threshold_dict, self.box_expand_size,
                             max_side=self.max_side, show=show,
                             divide_algo_param_dict=self.divide_algo_param_dict,
                             return_image=return_image, convert_to_gray=convert_to_gray,
                             check_validity=check_validity)


def resize_boxes(boxes, scale_x, scale_y):
    if boxes is None:
        return
    i1 = (1, 3)
    boxes[:, i1] = (boxes[:, i1] * scale_x).astype(np.uint16)
    i2 = (2, 4)
    boxes[:, i2] = (boxes[:, i2] * scale_y).astype(np.uint16)
