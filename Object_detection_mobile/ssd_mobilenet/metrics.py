from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tqdm

from pylibs.img_utils import show_img
from pylibs.json_utils import read_json_str
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe
from pylibs.rect_utils import UniversalRect
from pylibs.storage_utils import get_file_sharding
from pylibs.types import FilterFunction
from pylibs.vis_utils import draw_unirects_on_img
from pylibs.metrics_utils import detection_metrics


def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details, interpreter


def get_pred(input_details, output_details, interpreter, input_img):
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()
    boxes, classes, scores, num_boxes = [interpreter.get_tensor(output_details[i]['index']) for i in range(4)]
    return boxes[0], scores[0]


def run_model(dataset_dir_path: Path, df_name: str, model_path: str,
              img_dir: str, resize_shape: Tuple[int, int], iou_threshold: float = .5,
              num_filter: Optional[FilterFunction] = None, n=None,
              from_polygon: bool = False,
              img_format: Optional[str] = None,
              debug: bool = False):
    input_details, output_details, interpreter = load_model(model_path)

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = read_dataframe(df_path)
    df_size = len(df.index)
    new_df_size = n if n else df_size
    df = df.sample(new_df_size)
    df_size = new_df_size
    input_tensors, imgs, data_pairs = [], [], []
    for photo_id, row in tqdm.tqdm_notebook(df.iterrows(), total=df_size):
        if from_polygon:
            regions = read_json_str(row["regions"])
            if num_filter is not None:
                if not num_filter(len(regions)):
                    continue
            true_unirects = []
            for region in regions:
                shape_attrs = region['shape_attributes']
                ur = UniversalRect.from_via_polygon(shape_attrs['all_points_x'], shape_attrs['all_points_y'])
                true_unirects.append(ur)
        else:
            coords = read_json_str(row["tags"])["tags"]
            if num_filter is not None:
                if not num_filter(len(coords)):
                    continue
            true_unirects = [UniversalRect.from_coords_dict(c) for c in coords]
        if img_format:
            photo_path = get_file_sharding(img_dir, photo_id, '%s' + f'.{img_format}')
        else:
            photo_path = get_file_sharding(img_dir, photo_id)
        input_tensor, img = load_input_data(photo_path, resize_shape)
        # draw_unirects_on_img(img, true_unirects)
        # show_img(img)
        boxes, scores = get_pred(input_details, output_details, interpreter, input_tensor)
        pred_unirects = nn_boxes_to_unirects(boxes, img, iou_threshold, scores)
        pred_points = [u.min_max_points for u in pred_unirects]
        true_points = [u.min_max_points for u in true_unirects]
        if debug:
            res = detection_metrics([(true_points, pred_points)])
            if res['not_found_counter'] or res['extra_detection_counter'] or res['iou_l50']:
                print(res)
                draw_unirects_on_img(img, pred_unirects)
                show_img(img)
        data_pairs.append((true_points, pred_points))
    result = detection_metrics(data_pairs)
    print(result)


def nn_boxes_to_unirects(boxes, img, iou_threshold, scores):
    unirects = []
    h, w, _ = img.shape
    for box in (b for b, s in zip(boxes, scores) if s > iou_threshold):
        y1, x1, y2, x2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        unirects.append(UniversalRect((x1 + box_w / 2) * w, (y1 + box_h / 2) * h, box_w * w, box_h * h))
    return unirects


def load_input_data(photo_path, resize_shape):
    img = cv2.imread(str(photo_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(img, resize_shape)
    input_data = np.expand_dims(input_data, 0).astype(np.float32, copy=False)
    input_data = normalize_data(input_data)
    return input_data, img


def normalize_data(a: np.ndarray):
    return (2.0 / 255.0) * a - 1.0
