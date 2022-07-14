import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import tqdm

from price_detector.data_processing import defaultdict, read_pickle_local, \
    write_pickle_local
from price_detector.data_processing.utils import read_json, read_json_local, \
    write_array_local, \
    write_json_local
from tools.image_utils import is_image

SEED = 42
np.random.seed(SEED)
MAIN_DIR = Path('..')
MODELS_DIR = MAIN_DIR / 'models'
DATASETS_DIR = MAIN_DIR / 'datasets'
PNG_FORMAT = "%s.png"

VIA_SHAPE_NAME = "rect"


def prepare_for_labeling_via(img_save_dir: str, img_list_filename: str,
                             boxes_arrays_filename: str,
                             via_json_save_file: str):
    img_list = read_pickle_local(img_list_filename)
    img_save_path = DATASETS_DIR / img_save_dir
    img_save_path.mkdir(exist_ok=True)
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    via_dict = {}
    i = 0
    for boxes_list, img_list in tqdm.tqdm(zip(
            boxes_array_list.values(), img_list.values())):
        for c, (boxes, (tag_id, img)) in enumerate(zip(boxes_list, img_list)):
            filename = PNG_FORMAT % i
            img_path = img_save_path / filename
            cv2.imwrite(str(img_path), img)
            fsize = os.path.getsize(img_path)
            d = dict(filename=filename, size=fsize, regions=boxes_to_via_regions(boxes),
                     file_attributes=dict(cluster=c, tag_id=tag_id))
            if len(d['regions']) == 5 and c == 27:
                print("tag id:", tag_id, "i:", i, len(boxes), d["regions"])
            key = f"{filename}{fsize}"
            via_dict[key] = d
            i += 1
    write_json_local(via_dict, via_json_save_file)


def prepare_for_labeling_price_via(img_save_dir: str, img_list_filename: str,
                                   boxes_arrays_filename: str,
                                   via_json_save_file: str):
    img_list = read_pickle_local(img_list_filename)
    img_save_path = DATASETS_DIR / img_save_dir
    img_save_path2 = DATASETS_DIR / (img_save_dir + '_part2')
    img_save_path.mkdir(exist_ok=True)
    img_save_path2.mkdir(exist_ok=True)
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    via_dict, via_dict_part2 = {}, {}
    price_recognizer = PriceRecognizer()
    first_part = 250
    i = 0
    for boxes_list, img_list in tqdm.tqdm(zip(
            boxes_array_list.values(), img_list.values())):
        for c, (boxes, (tag_id, img)) in enumerate(zip(boxes_list, img_list)):
            filename = PNG_FORMAT % i
            img_path = (img_save_path if i <= first_part else img_save_path2) / filename
            cv2.imwrite(str(img_path), img)
            fsize = os.path.getsize(img_path)
            try:
                price = price_recognizer.recognize_float(img)
            except (KeyError, ValueError):
                price = .0
            d = dict(filename=filename, size=fsize, regions=boxes_to_via_regions(boxes),
                     file_attributes=dict(tag_id=tag_id, price=str(price)))
            key = f"{filename}{fsize}"
            if i <= first_part:
                via_dict[key] = d
            else:
                via_dict_part2[key] = d
            i += 1
    write_json_local(via_dict, via_json_save_file)
    write_json_local(via_dict_part2, via_json_save_file + '_part2')


def prepare_for_labeling_validation_via(tag_img_dir: str, photo_path: str,
                                        pricetag_boxes: str, min_w: int, min_h: int,
                                        via_json_save_file: str,
                                        modify_price_fun: Optional[Callable[[str],
                                                                            str]] = None):
    from price_detector.recognizer import PriceRecognizer
    pricetag_pred = read_pickle_local(pricetag_boxes)
    files = [os.path.join(r, f) for r, ds, fs in os.walk(photo_path) for f in fs if
             is_image(f)]
    files = sorted(files)
    img_list = [cv2.imread(str(img_name)) for img_name in files]

    img_save_path = DATASETS_DIR / tag_img_dir
    # img_save_path2 = DATASETS_DIR / (tag_img_dir + '_part2')
    img_save_path.mkdir(exist_ok=True)
    # img_save_path2.mkdir(exist_ok=True)
    via_dict, via_dict_part2 = {}, {}
    price_recognizer = PriceRecognizer(gpu=True)
    # first_part = 800
    # first_part = len(img_list) + 1

    photo_count, tag_count = 0, 0
    for img, pricetag_boxes in tqdm.tqdm(zip(img_list, pricetag_pred),
                                         total=len(pricetag_pred), disable=False):
        for box in pricetag_boxes:
            _, _, xmin, ymin, xmax, ymax = list(map(int, box))
            w, h = xmax - xmin, ymax - ymin
            # x, y = xmin + w // 2, ymin + h // 2
            # w -= 100
            # xmin, ymin = x - w // 2, y - h // 2
            # xmax, ymax = x + w // 2, y + h // 2
            if w <= min_w and h <= min_h:
                continue
            tag_count += 1

            crop = img[ymin:ymax, xmin:xmax]
            filename = PNG_FORMAT % tag_count
            save_path = img_save_path
            # save_path = img_save_path if tag_count <= first_part else img_save_path2
            photo_path = save_path / filename
            cv2.imwrite(str(photo_path), crop)
            fsize = os.path.getsize(photo_path)
            price = price_recognizer.recognize_float(crop)[0]
            if modify_price_fun is not None:
                price = modify_price_fun(price)
            d = dict(filename=filename, size=fsize,
                     file_attributes=dict(photo_id=photo_count, price=str(price)))
            key = f"{filename}{fsize}"
            via_dict[key] = d
            # if tag_count <= first_part:
            #     via_dict[key] = d
            # else:
            #     via_dict_part2[key] = d
        photo_count += 1
    print("num tags:", tag_count)
    print("photo num:", photo_count)
    write_json_local(via_dict, via_json_save_file)
    # if via_dict_part2:
    #     write_json_local(via_dict_part2, via_json_save_file + '_part2')


def boxes_to_via_regions(boxes_array):
    regions = []
    num_boxes = len(boxes_array)
    for box in boxes_array:
        _, x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        shape_attrs = dict(name=VIA_SHAPE_NAME, x=int(x1), y=int(y1),
                           width=int(width),
                           height=int(height))
        region = dict(shape_attributes=shape_attrs)
        regions.append(region)
    for i, region in enumerate(sorted(regions, key=lambda x: x["shape_attributes"]["x"])):
        if num_boxes == 4:
            if 0 <= i <= 1:
                reg_attrs = {"type": "rub"}
            else:
                reg_attrs = {"type": "kop"}
        elif num_boxes == 5:
            if 0 <= i <= 2:
                reg_attrs = {"type": "rub"}
            else:
                reg_attrs = {"type": "kop"}
        elif 2 <= num_boxes <= 3:
            reg_attrs = {"type": "rub"}
        else:
            reg_attrs = {"type": "noprice"}
        region["region_attributes"] = reg_attrs
    return regions


TYPE_CLASS_MAP = dict(noprice=0, rub=1, kop=2)


def parse_via_json_price(annotation_file1: str,
                         end_id: int,
                         annotation_file2: str,
                         save_array_name: str,
                         images_dir: str,
                         img_list_filename: str, boxes_array_filename: str):
    annotation = read_json(annotation_file1)
    annotation2 = read_json(annotation_file2)
    images_path = DATASETS_DIR / images_dir
    labels_list, img_list, boxes_list = [], [], []
    hashes = set()
    parse_annotation_dict_price_part(annotation, images_path, img_list,
                                     labels_list, boxes_list, hashes, end_id)
    parse_annotation_dict_price(annotation2, images_path,
                                img_list, labels_list, boxes_list, hashes)
    print(len(labels_list))
    write_array_local(save_array_name, np.array(labels_list))
    write_array_local(img_list_filename, np.array(img_list))
    write_array_local(boxes_array_filename, np.array(boxes_list))


def find_key(keys, prefix: str):
    for key in keys:
        if key.startswith(prefix):
            return key


def parse_annotation_dict_price_part(annotation_dict: dict, images_path: Path,
                                     img_list, label_list,
                                     boxes_list, hashes,
                                     end_id: int):
    keys = tuple(annotation_dict.keys())
    for i in range(end_id + 1):
        key = find_key(keys, str(i))
        v = annotation_dict[key]
        file_attrs = v["file_attributes"]
        tag_id = file_attrs["tag_id"]
        img_path = images_path / (PNG_FORMAT % tag_id)
        img = cv2.imread(str(img_path))
        img_hash = calc_img_hash(img)
        if img_hash in hashes:
            continue
        else:
            hashes.add(img_hash)
        img_list.append(img)
        boxes_array = regions_to_array(v)
        boxes_list.append(boxes_array)
        price = file_attrs["price"]
        label_list.append(float(price))


def regions_to_array(v):
    boxes = []
    for reg in v["regions"]:
        shape_attrrs = reg["shape_attributes"]
        x, y, width, height = shape_attrrs["x"], shape_attrrs["y"], \
                              shape_attrrs["width"], shape_attrrs["height"]
        box = [x, y, x + width, y + height]
        boxes.append(np.array(box))
    boxes_array = np.array(boxes)
    return boxes_array


def parse_annotation_dict_price(annotation_dict, images_path, img_list,
                                label_list, boxes_list, hashes):
    for v in annotation_dict.values():
        file_attrs = v["file_attributes"]
        tag_id = file_attrs["tag_id"]
        img_path = images_path / (PNG_FORMAT % tag_id)
        img = cv2.imread(str(img_path))
        img_hash = calc_img_hash(img)
        if img_hash in hashes:
            continue
        else:
            hashes.add(img_hash)
        img_list.append(img)
        price = file_attrs["price"]
        label_list.append(float(price))
        boxes_array = regions_to_array(v)
        boxes_list.append(boxes_array)


def calc_img_hash(img):
    return hash(tuple(tuple((tuple(y) for y in x)) for x in img))


def parse_via_json(annotation_file: str, save_pickle_file: str):
    annotation = read_json(annotation_file)
    save_boxes = defaultdict(list)
    for v in annotation.values():
        file_attrs = v["file_attributes"]
        cluster = file_attrs["cluster"]
        tag_id = file_attrs["tag_id"]
        num_boxes = len(v["regions"])
        tag_boxes = []
        for reg in v["regions"]:
            shape_attrrs = reg["shape_attributes"]
            x, y, width, height = shape_attrrs["x"], shape_attrrs["y"], \
                                  shape_attrrs["width"], shape_attrrs["height"]
            type = reg["region_attributes"]["type"]
            box = [TYPE_CLASS_MAP[type], x, y, x + width, y + height]
            tag_boxes.append(np.array(box))
        save_boxes[num_boxes].append((tag_id, cluster, np.array(tag_boxes)))
    for num_boxes in save_boxes.keys():
        save_boxes[num_boxes].sort(key=lambda x: x[1])

    write_pickle_local(save_pickle_file, save_boxes)


def prepare_via_json_price(annotation_file: str, src_img_dir,
                           dest_img_dir, save_annotation_file):
    img_dir1 = DATASETS_DIR / src_img_dir
    # suffix = '_part2'
    # img_dir2 = DATASETS_DIR / (src_img_dir + suffix)
    dest_img_dir = DATASETS_DIR / dest_img_dir
    dest_img_dir.mkdir(exist_ok=True)
    res_dict = {}
    annotation = read_json(annotation_file)
    # annotation2 = read_json(annotation_file1 + suffix)
    prepare_annotation_dict_validation(annotation, img_dir1, dest_img_dir, res_dict)
    # parse_annotation_dict_validation2(annotation2, img_dir2, dest_img_dir, res_dict)
    write_json_local(res_dict, save_annotation_file)


def prepare_annotation_dict_validation(annotation_dict: dict, img_dir, dest_dir,
                                       res_dict):
    # skip_vals = {'noprice', 'rub', 'kop'}
    skip_vals = {'noprice', 'kop', 'error'}
    for key, label_dict in annotation_dict.items():
        file_attrs = label_dict["file_attributes"]
        filename = label_dict["filename"]
        val = file_attrs["validate"]
        if val not in skip_vals:
            del file_attrs["validate"]
            res_dict[key] = label_dict
            shutil.copy(img_dir / filename, dest_dir)


def prepare_annotation_dict_validation2(annotation_dict: dict, img_dir, dest_dir,
                                        res_dict):
    for key, label_dict in annotation_dict.items():
        file_attrs = label_dict["file_attributes"]
        filename = label_dict["filename"]
        val = file_attrs["validate"]
        if val != 'noprice':
            del file_attrs["validate"]
            res_dict[key] = label_dict
            shutil.copy(img_dir / filename, dest_dir)


def parse_prices_verification(src_img_dir, via_json_file,
                              save_imgs_file1,
                              save_prices_file1,
                              save_imgs_file2, save_prices_file2):
    src_img_dir1 = DATASETS_DIR / src_img_dir
    suffix = '_part2'
    src_img_dir2 = DATASETS_DIR / (src_img_dir + suffix)
    annotations = read_json_local(via_json_file)
    annotations2 = read_json_local(via_json_file + suffix)
    img_list, price_list = [], []
    imgs_by_photo, prices_by_photo = defaultdict(list), defaultdict(list)
    parse_prices_verification_iter(annotations, src_img_dir1, img_list, price_list,
                                   imgs_by_photo, prices_by_photo)
    parse_prices_verification_iter(annotations2, src_img_dir2, img_list, price_list,
                                   imgs_by_photo, prices_by_photo)
    write_array_local(save_imgs_file1, np.array(img_list))
    write_array_local(save_prices_file1, np.array(price_list))
    write_pickle_local(save_imgs_file2, imgs_by_photo)
    write_pickle_local(save_prices_file2, prices_by_photo)


def parse_prices_verification_iter(ann_dict, src_img_dir,
                                   img_list, prices_list,
                                   img_dict, prices_dict):
    for key, label_dict in ann_dict.items():
        file_attrs = label_dict["file_attributes"]
        price = file_attrs["price"]
        if float(price) == 0.:
            continue
        filename = label_dict["filename"]
        photo_id = file_attrs["photo_id"]
        img_path = src_img_dir / filename
        img = cpp_lib.load(str(img_path))
        assert img is not None
        img_list.append(img)
        prices_list.append(price)
        img_dict[photo_id].append(img)
        prices_dict[photo_id].append(price)


def prepare_prices_verification(imgs_file: str, prices_file: str, img_save_dir: str,
                                first_size: int,
                                save_json_name: str):
    img_dict = read_pickle_local(imgs_file)
    prices_dict = read_pickle_local(prices_file)
    img_save_path = DATASETS_DIR / img_save_dir
    img_save_path2 = DATASETS_DIR / (img_save_dir + '_part2')
    img_save_path.mkdir(exist_ok=True)
    img_save_path2.mkdir(exist_ok=True)
    tag_count = 0
    via_dict, via_dict2 = {}, {}
    for photo_id, img_list in img_dict.items():
        for img, price in zip(img_list, prices_dict[photo_id]):
            filename = PNG_FORMAT % tag_count
            save_path = img_save_path if tag_count <= first_size else img_save_path2
            photo_path = save_path / filename
            cv2.imwrite(str(photo_path), img)
            fsize = os.path.getsize(photo_path)
            d = dict(filename=filename, size=fsize,
                     file_attributes=dict(photo_id=photo_id, price=str(price)))
            key = f"{filename}{fsize}"
            if tag_count <= first_size:
                via_dict[key] = d
            else:
                via_dict2[key] = d
            tag_count += 1
    format = 'zip'
    shutil.make_archive(img_save_path, format, img_save_path)
    shutil.make_archive(img_save_path2, format, img_save_path2)
    write_json_local(via_dict, save_json_name)
    write_json_local(via_dict2, save_json_name + '_part2')


def parse_via_json_price2(validation_annotation_file: str,
                          price_annotation_file1, price_annotation_file2,
                          src_img_dir, save_imgs_file, save_prices_file):
    val_anno1 = read_json(validation_annotation_file)
    price_anno1 = read_json(price_annotation_file1)
    price_anno2 = read_json(price_annotation_file2)
    src_img_dir1 = DATASETS_DIR / src_img_dir
    src_img_dir2 = DATASETS_DIR / (src_img_dir + '_part2')
    img_list = []
    prices_list = []

    parse_via_json_dict(img_list, prices_list, src_img_dir1, val_anno1,
                        accept_vals={'kop'})
    print(len(img_list))
    parse_via_json_dict(img_list, prices_list, src_img_dir1, price_anno1, src_img_dir2)
    print(len(img_list))
    parse_via_json_dict(img_list, prices_list, src_img_dir1, price_anno2)
    print(len(img_list))

    write_array_local(save_imgs_file, np.array(img_list))
    write_array_local(save_prices_file, np.array(prices_list))


def parse_via_json_dict(img_list, prices_list, src_img_dir1, val_anno1,
                        src_img_dir2=None,
                        accept_vals=None):
    for key, label_dict in val_anno1.items():
        file_attrs = label_dict["file_attributes"]
        filename = label_dict["filename"]
        if accept_vals is not None:
            val = file_attrs["validate"]
            if val not in accept_vals:
                continue
        img_path = src_img_dir1 / filename
        # img = cpp_lib.load(str(img_path))
        if not img_path.exists():
            if src_img_dir2 is not None:
                img_path = src_img_dir2 / filename
                assert img_path.exists()
        img = cv2.imread(str(img_path))
        assert img is not None
        img_list.append(img)
        price = file_attrs["price"]
        prices_list.append(price)


def parse_via_json_price3(validation_annotation_file: str,
                          price_annotation_file1, price_annotation_file2,
                          src_img_dir, save_imgs_file, save_prices_file):
    val_anno1 = read_json(validation_annotation_file)
    price_anno1 = read_json(price_annotation_file1)
    price_anno2 = read_json(price_annotation_file2)
    src_img_dir1 = DATASETS_DIR / src_img_dir
    src_img_dir2 = DATASETS_DIR / (src_img_dir + '_part2')
    img_dict = defaultdict(list)
    prices_dict = defaultdict(list)

    parse_via_json_dict2(img_dict, prices_dict, src_img_dir1, val_anno1,
                         accept_vals={'kop'})
    print(len(img_dict))
    parse_via_json_dict2(img_dict, prices_dict, src_img_dir1, price_anno1, src_img_dir2)
    print(len(img_dict))
    parse_via_json_dict2(img_dict, prices_dict, src_img_dir1, price_anno2)
    print(len(img_dict))

    write_pickle_local(save_imgs_file, img_dict)
    write_pickle_local(save_prices_file, prices_dict)


def parse_via_json_dict2(img_dict, prices_dict, src_img_dir1, val_anno1,
                         src_img_dir2=None,
                         accept_vals=None):
    for key, label_dict in val_anno1.items():
        file_attrs = label_dict["file_attributes"]
        photo_id = file_attrs["photo_id"]
        filename = label_dict["filename"]
        if accept_vals is not None:
            val = file_attrs["validate"]
            if val not in accept_vals:
                continue
        img_path = src_img_dir1 / filename
        # img = cpp_lib.load(str(img_path))
        if not img_path.exists():
            if src_img_dir2 is not None:
                img_path = src_img_dir2 / filename
                assert img_path.exists()
        img = cv2.imread(str(img_path))
        assert img is not None
        img_dict[photo_id].append(img)
        price = file_attrs["price"]
        prices_dict[photo_id].append(price)


def parse_via_json_price4(ann_dict_file,
                          src_img_dir,
                          save_imgs_file, save_prices_file,
                          save_imgs_dict_file, save_prices_dict_file):
    price_anno1 = read_json_local(ann_dict_file)
    src_img_dir = DATASETS_DIR / src_img_dir
    img_list = []
    prices_list = []
    img_dict = defaultdict(list)
    prices_dict = defaultdict(list)
    for key, label_dict in price_anno1.items():
        file_attrs = label_dict["file_attributes"]
        price = file_attrs["price"]
        if float(price) == 0.:
            continue
        filename = label_dict["filename"]
        photo_id = file_attrs["photo_id"]
        img_path = src_img_dir / filename
        img = cpp_lib.load(str(img_path))
        # img = cv2.imread(str(img_path))
        assert img is not None
        img_list.append(img)
        prices_list.append(price)
        img_dict[photo_id].append(img)
        price = file_attrs["price"]
        prices_dict[photo_id].append(price)

    write_array_local(save_imgs_file, np.array(img_list))
    write_array_local(save_prices_file, np.array(prices_list))
    write_pickle_local(save_imgs_dict_file, img_dict)
    write_pickle_local(save_prices_dict_file, prices_dict)
