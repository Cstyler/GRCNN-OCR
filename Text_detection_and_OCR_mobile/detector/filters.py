from typing import Tuple

import numpy as np

from price_detector.detector.box_utils import box_area, box_areas, iou


def filter_boxes_by_area(boxes: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    new_boxes = []
    for box in boxes:
        area = box_area(box)
        if min_area < area and area < max_area:
            new_boxes.append(box)
    return np.array(new_boxes)


def filter_boxes_by_ar(boxes: np.ndarray, min_ar: float, max_ar: float) -> np.ndarray:
    new_boxes = []
    for box in boxes:

        box_h = box[2] - box[0]
        box_w = box[3] - box[1]
        if not box_h or not box_w:
            continue

        ar = box_h / box_w
        if min_ar < ar and ar < max_ar:
            new_boxes.append(box)
    return np.array(new_boxes)


def filter_boxes_by_same(boxes: np.ndarray) -> np.ndarray:
    boxes_left = np.copy(boxes)
    boxes_result = []
    while len(boxes_left):
        box = boxes_left[0]
        boxes_result.append(box)

        delete_indices = []
        for i, box_i in enumerate(boxes_left):
            if np.array_equal(box, box_i):
                delete_indices.append(i)
        boxes_left = np.delete(boxes_left, delete_indices, axis=0)

    return np.array(boxes_result)


def region_area(region: np.ndarray) -> float:
    n = len(region)
    area = .0
    for i in range(n):
        j = (i + 1) % n
        reg1 = region[i]
        reg2 = region[j]
        area += reg1[0] * reg2[1]
        area -= reg2[0] * reg1[1]
    return abs(area) / 2.


def filter_boxes_by_region_and_box_area(boxes: np.ndarray,
                                        regions: np.ndarray,
                                        area_ratio_threshold: float) -> np.ndarray:
    new_boxes = []
    for box, region in zip(boxes, regions):
        ra = region_area(region)
        ba = box_area(box)

        if max(ra, ba) == 0:
            continue

        if min(ra, ba) / max(ra, ba) > area_ratio_threshold:
            new_boxes.append(box)
    return np.array(new_boxes)


def filter_boxes_by_same_digits(boxes: np.ndarray, classes: np.ndarray,
                                iou_threshold: float) -> Tuple[
    np.ndarray, np.ndarray]:
    boxes_left = np.copy(boxes)
    classes_left = np.copy(classes)

    areas = box_areas(boxes).argsort(kind='stable')
    boxes_left = boxes_left[areas]
    classes_left = classes_left[areas]

    boxes_result = []
    classes_result = []
    while len(boxes_left):
        box = boxes_left[-1]
        cl = classes_left[-1]
        boxes_result.append(box)
        classes_result.append(cl)

        delete_indices = []
        for i, box_i in enumerate(boxes_left):
            if iou(box_i, box, mode='min') >= iou_threshold:
                delete_indices.append(i)

        boxes_left = np.delete(boxes_left, delete_indices, axis=0)
        classes_left = np.delete(classes_left, delete_indices, axis=0)

    return np.array(boxes_result)[::-1], np.array(classes_result)[::-1]


def filter_boxes_by_same_iou(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    boxes_left = np.copy(boxes)
    areas = box_areas(boxes)
    boxes_left = boxes_left[areas.argsort(kind='stable')]

    boxes_result = []
    while len(boxes_left):
        box = boxes_left[-1]
        boxes_result.append(box)

        delete_indices = []
        for i, box_i in enumerate(boxes_left):
            iou_res = iou(box_i, box, mode='union')
            if iou_res >= iou_threshold:
                delete_indices.append(i)
        boxes_left = np.delete(boxes_left, delete_indices, axis=0)

    return np.array(boxes_result)[::-1]


def filter_boxes_by_iou(boxes: np.ndarray,
                        iou_threshold: float,
                        area_ratio_threshold: float) -> np.ndarray:
    boxes_left = np.copy(boxes)
    maxima = []
    while len(boxes_left):
        index = -1
        box = boxes_left[index]
        maxima.append(box)
        boxes_left = np.delete(boxes_left, index, axis=0)
        keep_indices = []
        for i, boxi in enumerate(boxes_left):
            rate = iou(boxi, box, mode='min')
            ba = box_area(box)
            bai = box_area(boxi)

            if max(ba, bai) == 0:
                continue

            if rate < iou_threshold or min(ba, bai) / max(ba, bai) < area_ratio_threshold:
                keep_indices.append(i)
        boxes_left = boxes_left[keep_indices]

    return np.array(maxima)


def filter_small_boxes(boxes: np.ndarray, area: float,
                       area_ratio_threshold: float) -> np.ndarray:
    if not len(boxes) or not area:
        return boxes

    if box_areas(boxes).mean() / area < area_ratio_threshold:
        return np.array([])
    return boxes
