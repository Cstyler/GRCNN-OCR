from pathlib import Path

import cv2

from pylibs import img_utils, jpeg_utils, pandas_utils
from pylibs.cutter import cut_segment_from_img
from pylibs.json_utils import json
from pylibs.pandas_utils import DF_FILE_FORMAT
from pylibs.rect_utils import UniversalRect
from pylibs.storage_utils import get_file_sharding, save_file_sharding
from tqdm.notebook import tqdm
import numpy as np


def crop_images(dataset_dir_path: str, df_name: str,
                img_dir: str, train_df_name: str, save_df_name: str, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    processed_img_dir_path = dataset_dir_path / "processed_images"
    img_resized_dir_path = dataset_dir_path / img_dir
    img_resized_dir_path.mkdir(exist_ok=True)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    indices = []
    for tag_id, row in df.iterrows():
        img_path = get_file_sharding(processed_img_dir_path, tag_id)
        img = jpeg_utils.read_jpeg(img_path)
        if row.rect:
            rect_dict = json.loads(row.rect)
            unirect = UniversalRect.from_coords_dict(rect_dict)
            rectified_img = cut_segment_from_img(img, unirect)
            if debug:
                img_utils.show_img(rectified_img, (3, 5))
            else:
                save_img_path = save_file_sharding(img_resized_dir_path, img_path)
                jpeg_utils.write_jpeg(save_img_path, rectified_img)
            indices.append(tag_id)
    train_df_path = dataset_dir_path / (DF_FILE_FORMAT % train_df_name)
    train_df = pandas_utils.read_dataframe(train_df_path)
    new_df = train_df.loc[indices]
    new_df_path = dataset_dir_path / (DF_FILE_FORMAT % save_df_name)
    pandas_utils.write_dataframe(new_df, new_df_path, index=True)


def resize_images(dataset_dir_path: str, df_name: str, img_dir: str, save_img_dir: str,
                  width: int, height: int, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    source_img_dir = dataset_dir_path / img_dir
    img_resized_dir_path = dataset_dir_path / save_img_dir
    img_resized_dir_path.mkdir(exist_ok=True)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)

    shape = (width, height)
    for tag_id in df.index:
        img_path = get_file_sharding(source_img_dir, tag_id)
        save_img_path = save_file_sharding(img_resized_dir_path, img_path)
        if save_img_path.exists():
            continue
        try:
            img = jpeg_utils.read_jpeg(img_path)
        except (OSError, IOError):
            continue
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        if debug:
            img_utils.show_img(img, (3, 5))
        else:
            jpeg_utils.write_jpeg(save_img_path, img)


def resize_images_padding(dataset_dir_path: str, df_name: str, img_dir: str, save_img_dir: str,
                          width: int, height: int, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    source_img_dir = dataset_dir_path / img_dir
    img_resized_dir_path = dataset_dir_path / save_img_dir
    img_resized_dir_path.mkdir(exist_ok=True)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)

    index = df.index
    for tag_id in tqdm(index, total=len(index), smoothing=.01):
        img_path = get_file_sharding(source_img_dir, tag_id)
        save_img_path = save_file_sharding(img_resized_dir_path, img_path)
        if save_img_path.exists():
            continue
        try:
            img = jpeg_utils.read_jpeg(img_path)
        except (OSError, IOError):
            continue
        h, w, _ = img.shape
        target_w = int(height / h * w)
        shape = target_w, height
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        after_pad = width - target_w
        if after_pad > 0:
            pad_width = [(0, 0), (0, after_pad)]
            # color = get_optimal_pad_value(img)
            color = (0, 0, 0)
            img = [np.pad(img[..., c], pad_width, 'constant', constant_values=color[c]) for c in range(3)]
            img = np.stack(img, -1)
        else:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        assert img.shape == (height, width, 3)
        if debug:
            img_utils.show_img(img, (4, 1))
        else:
            jpeg_utils.write_jpeg(save_img_path, img)


def remove_old_imgs(dataset_dir_path: str, df_name: str,
                    old_df_name: str, img_dir_pth: str, debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    img_dir_pth = Path(img_dir_pth)
    old_df_path = dataset_dir_path / (DF_FILE_FORMAT % old_df_name)
    old_df = pandas_utils.read_dataframe(old_df_path)
    new_df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    new_df = pandas_utils.read_dataframe(new_df_path)
    old_df_index = old_df.index
    new_df_index = new_df.index
    count = 0
    for tag_id in old_df_index:
        if tag_id not in new_df_index:
            img_path = get_file_sharding(img_dir_pth, tag_id)
            if img_path.exists():
                count += 1
                if not debug:
                    img_path.unlink()
    if debug:
        print("%s images need to be deleted" % count)
    else:
        print("%s images were deleted" % count)
