import math
from typing import List, Optional, Union

import numpy as np

from price_detector.detector.utils import show_boxes_on_image


def boxes_from_regions(regions: np.ndarray) -> np.ndarray:
    boxes = []
    for region in regions:
        xmin = region[:, 0].min()
        ymin = region[:, 1].min()
        xmax = region[:, 0].max()
        ymax = region[:, 1].max()
        boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)


def intersection_area(boxA: np.ndarray, boxB: np.ndarray) -> np.int:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea


def box_area(box: Union[np.ndarray, List[float]]) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def box_center_x(box: np.ndarray) -> np.float:
    return (box[0] + box[2]) / 2


def box_aspect_ratio(box: np.ndarray) -> np.float:
    return (box[2] - box[0]) / (box[3] - box[1])


def iou(boxA: np.ndarray, boxB: np.ndarray, mode: str) -> np.float:
    """
    mode: 'min' or 'union'
    """
    interArea = intersection_area(boxA, boxB)
    boxAArea = box_area(boxA)
    boxBArea = box_area(boxB)
    if mode == 'min':
        union_areas = min(boxAArea, boxBArea)
    elif mode == 'union':
        union_areas = boxAArea + boxBArea - interArea
    else:
        raise ValueError("Wrong mode: %s" % mode)

    if union_areas <= 0:
        return 0.
    return interArea / union_areas


def calc_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    n = len(boxes)
    iou_matrix = np.empty((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                iou_matrix[i, j] = 1
            else:
                res_iou = iou(boxes[i], boxes[j], mode='union')
                iou_matrix[i, j] = res_iou
                iou_matrix[j, i] = res_iou
    return iou_matrix


def box_areas(boxes: np.ndarray) -> np.ndarray:
    return np.array([box_area(box) for box in boxes])


def box_aspect_ratios(boxes: np.ndarray) -> np.ndarray:
    return np.array([box_aspect_ratio(box) for box in boxes])


def expand_boxes(boxes: np.ndarray, expand_size: int, h: int, w: int) -> np.ndarray:
    boxes = np.copy(boxes)
    for i, box in enumerate(boxes):
        xmin, ymin = max(box[0] - expand_size, 0), max(box[1] - expand_size, 0)
        xmax, ymax = min(box[2] + expand_size, w), min(box[3] + expand_size, h)
        boxes[i] = [xmin, ymin, xmax, ymax]
    return boxes


# Проверяем, что координаты (икс либо игрек) ббокса 1 (min1, max1) удовлетворяют условию
# нахождения внутри ббокса 2 (min2, max2). по игреку проверяем, что высота составляет
# некоторую часть высоты большого ббокса (min_dist_ratio/max_dist_ratio). по иксу
# проверяем что ширина ббокса1 не больше какой-то части ширины большого ббокса(
# min_dist_ratio/max_dist_ratio). По координатам ббоксов  проверяем, что ббокс
# находится внутри, либо не должен   заезжать за край ббокса больше чем на какой-то
# процент(rel_tol)
def box_inside_predicate(min1: int, min2: int, max1: int, max2: int,
                         rel_tol: float, min_dist_ratio: float, max_dist_ratio: float):
    dist_ratio = (max2 - min2) / (max1 - min1)
    return (min1 < min2 or math.isclose(min1, min2, rel_tol=rel_tol)) and \
           (max1 > max2 or math.isclose(max1, max2, rel_tol=rel_tol)) and \
           min_dist_ratio < dist_ratio < max_dist_ratio


# дочерние ббоксы часто слипаются вместе, из-за чего алгоритм добавления ббоксов
# неправильно работает. Удаляем центры которых ббоксы находятся слишком близко друг к
# другу: делим расстояние между центрами на среднюю ширину ббоксов и проверяем что это
# число лежит в некотором интервале.
def filter_boxes_by_center_dist(boxes: np.ndarray, left_thr: float,
                                right_thr: float) -> np.ndarray:
    boxes_left = np.copy(boxes)
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_width_mean = box_widths.mean()
    boxes_result = []
    while len(boxes_left):
        box = boxes_left[-1]
        boxes_result.append(box)
        delete_indices = []
        center_x = box_center_x(box)
        for i, box_i in enumerate(boxes_left[:-1]):
            center_dist = abs(box_center_x(box_i) - center_x)
            dist_ratio = center_dist / box_width_mean
            if not (left_thr < dist_ratio < right_thr):
                delete_indices.append(i)
        delete_indices.append(len(boxes_left) - 1)
        boxes_left = np.delete(boxes_left, delete_indices, axis=0)
    return np.array(boxes_result)


# в данных есть особенность - мсер детектирует большие ббоксы, которые содержат ббоксы
# цифр, но не все цифры внутри ббокса распознаны. эта функция находит такие ббоксы и их
# дочерние ббоксы и вставляет новые ббоксы в свободные места внутри родительского ббокса.
def divide_boxes_that_contain_another_boxes(boxes: np.ndarray,
                                            param_dict: dict,
                                            img: Optional[
                                                np.ndarray] = None) -> np.ndarray:
    show = img is not None
    boxes_left = boxes.copy()
    areas = box_areas(boxes)
    boxes_left = boxes_left[areas.argsort(kind='stable')]
    # коэффициенты связанные с предикатом того что бокс дочерний.
    x_coord_rel_tol, \
    y_coord_rel_tol = param_dict["x_coord_rel_tol"], param_dict["y_coord_rel_tol"]
    min_width_ratio, \
    max_width_ratio = param_dict["min_width_ratio"], param_dict["max_width_ratio"]
    min_height_ratio, \
    max_height_ratio = param_dict["min_height_ratio"], param_dict["max_height_ratio"]
    # в алгоритме считается коэффициент: свободное расстояние между ббоксами делённое на
    # среднюю длину ббокса. чтобы понять сколько ббоксов можно вставить,
    # нужно проверить в каком интервале находится этот коэффициент, если в
    # (free_space_width_for_one_box_left_thr, free_space_width_for_one_box_right_thr) то 1 ббокс, если в
    # (free_space_width_for_one_box_right_thr, right_width_threshold2), то 2
    free_space_width_for_one_box_left_thr, \
    free_space_width_for_one_box_right_thr = param_dict[
                                                 "free_space_width_for_one_box_left_thr"], \
                                             param_dict[
                                                 "free_space_width_for_one_box_right_thr"]
    free_space_width_for_two_boxes_right_thr = param_dict[
        "free_space_width_for_two_boxes_right_thr"]
    min_box_dist, \
    max_box_dist = param_dict["min_box_dist"], \
                   param_dict["max_box_dist"]
    # добавочное расстояние между цифрами
    box_padding = 1
    double_box_padding = 2 * box_padding
    triple_box_padding = 3 * box_padding
    union_boxes, boxes_result = [], []
    # максимальное кол-во ббоксов внутри родительского 2, потому что нужно ещё иметь
    # место для третьего ббокса.
    MAX_CHILD_BOXES_NUM = 3
    while len(boxes_left):
        boxes_left = find_parent_child_boxes(boxes_left, boxes_result, union_boxes,
                                             min_width_ratio, max_height_ratio,
                                             max_width_ratio, min_height_ratio,
                                             min_box_dist, max_box_dist,
                                             x_coord_rel_tol, y_coord_rel_tol,
                                             MAX_CHILD_BOXES_NUM)
    for box, child_boxes in union_boxes:
        xmin1, ymin1, xmax1, ymax1 = box
        child_boxes_widths = child_boxes[:, 2] - child_boxes[:, 0]
        digit_box_width_mean = child_boxes_widths.mean()
        if len(child_boxes) == 1:
            child = child_boxes[0]
            child_xmin1 = child[0]
            child_xmax1 = child[2]
            left_space_width = child_xmin1 - xmin1
            right_space_width = xmax1 - child_xmax1
            # если слева и справа есть место для ббокса вставляем 2 ббокса
            if free_space_width_for_one_box_left_thr < (
                    left_space_width - double_box_padding) / digit_box_width_mean < \
                    free_space_width_for_one_box_right_thr and \
                    free_space_width_for_one_box_left_thr < (
                    right_space_width - double_box_padding) / digit_box_width_mean < \
                    free_space_width_for_one_box_right_thr:
                new_bbox_width1 = left_space_width - double_box_padding
                new_bbox1 = np.asarray(
                    [xmin1 + box_padding, ymin1 + box_padding,
                     xmin1 + box_padding + new_bbox_width1,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox1)
                new_bbox_width2 = right_space_width - double_box_padding
                new_bbox2 = np.asarray(
                    [child_xmax1 + box_padding, ymin1 + box_padding,
                     child_xmax1 + new_bbox_width2,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox2)
                if show:
                    print("Find, insert 2 boxes in left/right", box, child_boxes,
                          "new bbox1:", new_bbox1, "new bbox2:", new_bbox2)
                    show_boxes_on_image(np.asarray([box] + list(child_boxes)), img)
                    show_boxes = np.asarray(
                        [box, new_bbox1, new_bbox2] + list(child_boxes))
                    show_boxes_on_image(show_boxes, img)
            #  если слева есть место на 1 ббокс, вставляем его туда
            elif free_space_width_for_one_box_left_thr < (
                    left_space_width - double_box_padding) / digit_box_width_mean < \
                    free_space_width_for_one_box_right_thr:
                child_xmin1 = child[0]
                new_bbox = np.asarray([xmin1 + box_padding, ymin1 + box_padding,
                                       child_xmin1 - box_padding,
                                       ymax1 - box_padding])
                boxes_result.append(new_bbox)
                if show:
                    print("Insert 1 bbox left", box, child_boxes, "new bbox:", new_bbox)
                    show_boxes_on_image(np.asarray([box] + list(child_boxes)), img)
                    show_boxes_on_image(np.asarray([box, new_bbox] + list(
                        child_boxes)), img)
            # если справа есть место на 1 ббокс, вставляем его туда
            elif free_space_width_for_one_box_left_thr < (
                    right_space_width - double_box_padding) / digit_box_width_mean < \
                    free_space_width_for_one_box_right_thr:
                child_xmax1 = child[2]
                new_bbox = np.asarray(
                    [child_xmax1 + box_padding, ymin1 + box_padding,
                     xmax1 - box_padding,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox)
                if show:
                    print("Insert 1 bbox right", box, child_boxes, "new bbox:", new_bbox)
                    show_boxes_on_image(np.asarray([box] + list(child_boxes)), img)
                    show_boxes = np.asarray([box, new_bbox] + list(child_boxes))
                    show_boxes_on_image(show_boxes, img)
            # если слева есть место для двух ббоксов, вставляем туда их
            elif free_space_width_for_one_box_right_thr < (
                    left_space_width - triple_box_padding) / digit_box_width_mean < \
                    free_space_width_for_two_boxes_right_thr:
                new_bbox_width = int((left_space_width - triple_box_padding) / 2)
                xmax_bbox1 = xmin1 + box_padding + new_bbox_width
                new_bbox1 = np.asarray(
                    [xmin1 + box_padding, ymin1 + box_padding,
                     xmax_bbox1,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox1)
                xmin_bbox2 = xmax_bbox1 + box_padding
                new_bbox2 = np.asarray(
                    [xmin_bbox2, ymin1 + box_padding,
                     xmin_bbox2 + new_bbox_width,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox2)
                if show:
                    print("Find, insert 2 boxes in left", box, child_boxes,
                          "new bbox:", new_bbox1, "new bbox2:", new_bbox2)
                    show_boxes_on_image(np.asarray([box] + list(child_boxes)), img)
                    show_boxes = np.asarray(
                        [box, new_bbox1, new_bbox2] + list(child_boxes))
                    show_boxes_on_image(show_boxes, img)
            # если справа есть место для двух ббоксов, вставляем туда их
            elif free_space_width_for_one_box_right_thr < (
                    right_space_width - triple_box_padding) / digit_box_width_mean < \
                    free_space_width_for_two_boxes_right_thr:
                new_bbox_width = int((right_space_width - triple_box_padding) / 2)
                xmax_bbox1 = child_xmax1 + box_padding + new_bbox_width
                new_bbox1 = np.asarray(
                    [child_xmax1 + box_padding, ymin1 + box_padding,
                     xmax_bbox1,
                     ymax1 - box_padding])
                xmin_bbox2 = xmax_bbox1 + box_padding
                new_bbox2 = np.asarray(
                    [xmin_bbox2, ymin1 + box_padding,
                     xmin_bbox2 + new_bbox_width,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox1)
                boxes_result.append(new_bbox2)
                if show:
                    print("Find, insert 2 boxes in right", box, child_boxes,
                          "new bbox1:", new_bbox1, "new bbox2:", new_bbox2)
                    show_boxes_on_image(np.asarray([box] + list(child_boxes)), img)
                    show_boxes = np.asarray(
                        [box, new_bbox1, new_bbox2] + list(child_boxes))
                    show_boxes_on_image(show_boxes, img)
            else:
                # print("Append to result parent:", box)
                boxes_result.append(box)
        elif len(child_boxes) == 2:
            child_boxes = child_boxes[child_boxes[:, 0].argsort()]
            left_child = child_boxes[0]
            left_child_xmin1, left_child_xmax1 = left_child[0], left_child[2]
            right_child = child_boxes[-1]
            right_child_xmin1, right_child_xmax1 = right_child[0], right_child[2]
            left_space_width = left_child_xmin1 - xmin1
            right_space_width = xmax1 - right_child_xmax1
            between_space_width = right_child_xmin1 - left_child_xmax1
            # если слева есть место, вставляем туда ббокс
            if free_space_width_for_one_box_left_thr < (
                    left_space_width - double_box_padding) / digit_box_width_mean < \
                    free_space_width_for_one_box_right_thr:
                new_bbox = np.asarray([xmin1 + box_padding, ymin1 + box_padding,
                                       left_child_xmin1 - box_padding,
                                       ymax1 - box_padding])
                boxes_result.append(new_bbox)
                if show:
                    print("Insert 1 bbox left. Child box 2", box, child_boxes,
                          "new bbox:", new_bbox)
            # если справа есть место, вставляем туда ббокс
            elif free_space_width_for_one_box_left_thr < (
                    right_space_width - double_box_padding) / digit_box_width_mean < \
                    free_space_width_for_one_box_right_thr:
                new_bbox = np.asarray(
                    [right_child_xmax1 + box_padding, ymin1 + box_padding,
                     xmax1 - box_padding,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox)
                if show:
                    print("Insert 1 bbox right. Child box 2", box, child_boxes,
                          "new bbox:", new_bbox)
                    show_boxes_on_image(np.asarray([box] + list(child_boxes)), img)
                    show_boxes_on_image(np.asarray([box, new_bbox] + list(
                        child_boxes)), img)
            # если посередине между ббоксами есть место, вставляем туда ббокс
            elif free_space_width_for_one_box_left_thr < (
                    between_space_width - double_box_padding) / digit_box_width_mean < \
                    free_space_width_for_one_box_right_thr:
                new_bbox = np.asarray(
                    [left_child_xmax1 + box_padding, ymin1 + box_padding,
                     right_child_xmin1 - box_padding,
                     ymax1 - box_padding])
                boxes_result.append(new_bbox)
                if show:
                    print("Insert 1 bbox in between. Child box 2", box, child_boxes,
                          "new bbox:", new_bbox)
                    show_boxes_on_image(np.asarray([box] + list(child_boxes)), img)
                    show_boxes_on_image(np.asarray([box, new_bbox] + list(
                        child_boxes)), img)
            else:
                boxes_result.append(box)
        else:
            boxes_result.append(box)
    return np.asarray(boxes_result)


def find_parent_child_boxes(boxes_left, boxes_result, union_boxes, min_width_ratio,
                            max_height_ratio, max_width_ratio, min_height_ratio,
                            min_box_dist, max_box_dist, x_coord_rel_tol,
                            y_coord_rel_tol, MAX_CHILD_BOXES_NUM):
    box = boxes_left[-1]
    xmin1, ymin1, xmax1, ymax1 = box
    delete_indices = []
    child_boxes = []
    for i, box_i in enumerate(boxes_left[:-1]):
        xmin2, ymin2, xmax2, ymax2 = box_i

        pred1 = box_inside_predicate(xmin1, xmin2, xmax1, xmax2, x_coord_rel_tol,
                                     min_width_ratio, max_width_ratio)
        pred2 = box_inside_predicate(ymin1, ymin2, ymax1, ymax2, y_coord_rel_tol,
                                     min_height_ratio, max_height_ratio)
        if pred1 and pred2:
            child_boxes.append(box_i)
            boxes_result.append(box_i)
            delete_indices.append(i)
    delete_indices.append(len(boxes_left) - 1)
    boxes_left = np.delete(boxes_left, delete_indices, axis=0)
    if child_boxes:
        child_boxes = np.asarray(child_boxes)
        child_boxes = filter_boxes_by_center_dist(child_boxes,
                                                  min_box_dist,
                                                  max_box_dist)
    if 0 < len(child_boxes) < MAX_CHILD_BOXES_NUM:
        union_boxes.append((box, child_boxes))
    else:
        boxes_result.append(box)
    return boxes_left
