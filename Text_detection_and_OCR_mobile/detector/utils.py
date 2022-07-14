from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_regions(regions: np.ndarray, img: np.ndarray):
    pts = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    blue_color = (0, 0, 255)
    img = img.copy()
    cv2.polylines(img, pts, 1, blue_color, 1)
    plt.imshow(img[..., ::-1])
    plt.show()


def show_img(img: np.ndarray):
    plt.figure(figsize=(7, 7), dpi=70)
    plt_imshow(img)
    plt.show()
    plt.close()


def plt_imshow(img):
    if img.ndim == 3:
        img = img[..., ::-1]
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')


def resize_boxes(img: np.ndarray, boxes: np.ndarray, max_side: int, copy=False):
    if copy:
        boxes = np.array([x.copy() for x in boxes])
    img_h, img_w, _ = img.shape
    if img_h > img_w:
        h = max_side
        w = int(h / img_h * img_w)
    else:
        w = max_side
        h = int(w / img_w * img_h)
    if len(boxes):
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * img_w / w).astype(np.uint16)
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * img_h / h).astype(np.uint16)


def show_boxes_on_image(boxes: np.ndarray, img: np.ndarray,
                        classes: Optional[Union[List[int]]] = None,
                        classes_map: Optional[Dict[int, int]] = None,
                        write_to_file: Optional[str] = None):
    fig = plt.figure(figsize=(7, 7), dpi=70)
    fig.patch.set_facecolor((1., 1., 1.))
    plt_imshow(img)

    current_axis = plt.gca()
    text_bbox_params = {'facecolor': 'red', 'alpha': 1.}
    for i, boxes in enumerate(boxes):
        boxes_len = len(boxes)
        if boxes_len == 4:
            (xmin, ymin, xmax, ymax) = boxes
            class_ = None
        elif boxes_len == 5:
            (class_, xmin, ymin, xmax, ymax) = boxes
        else:
            raise ValueError("Wrong boxes len:", boxes_len)
        w, h = xmax - xmin, ymax - ymin
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), w, h, color='red', fill=False, linewidth=1))

        if classes is not None:
            class_ = classes[i]
        if classes_map is not None:
            class_ = classes_map[class_]
        if class_ is not None:
            current_axis.text(xmin, ymin, str(class_), color='white',
                              bbox=text_bbox_params)
    plt.tight_layout()
    if write_to_file is None:
        plt.show()
    else:
        plt.savefig(write_to_file)
    plt.close()
