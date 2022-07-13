from pathlib import Path
from shutil import copyfile
from typing import Iterable

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm

from pylibs.jpeg_utils import JPEG_SUFFIX, read_jpeg, write_jpeg
from pylibs.json_utils import read_json_str
from pylibs.numpy_utils import NPY_SUFFIX, write_array
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe, write_dataframe
from pylibs.rect_utils import UniversalRect
from pylibs.storage_utils import get_file_sharding

WHITE_COLOR = 255


def resize(img: np.ndarray, mask: np.ndarray, max_side: int, inter: int = cv2.INTER_CUBIC):
    h, w, _ = img.shape
    max_ = max(h, w)
    h = h * max_side // max_
    w = w * max_side // max_
    dsize = (w, h)
    return cv2.resize(img, dsize, interpolation=inter), \
           cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)


def get_path_pairs(base_dir: Path, subdir1: str = 'images', subdir_ext1: str = JPEG_SUFFIX,
                   subdir2: str = 'masks', subdir_ext2: str = NPY_SUFFIX):
    # generator, which returns path for image and corresponding path for json
    images_dir = base_dir / subdir1
    images_names = [Path(p.name) for p in images_dir.glob(f'*{subdir_ext1}')]
    for image_name in images_names:
        json_name = image_name.with_suffix(subdir_ext2)
        json_path = base_dir / subdir2 / json_name
        image_path = images_dir / image_name
        yield image_path, json_path


def transfer_data(imgs: Iterable[Path], masks: Iterable[Path], img_dir: Path, label_dir: Path):
    for image_path, mask_path in tqdm(zip(imgs, masks), total=len(imgs)):
        if image_path.is_file():
            stem = Path(Path(image_path).stem)
            copyfile(image_path, img_dir / stem.with_suffix(JPEG_SUFFIX))
            copyfile(mask_path, label_dir / stem.with_suffix(NPY_SUFFIX))


def prepare_for_train(output_dir: Path, cache_base_dir: Path, cache_dir_name: str, test_size: float):
    cache_dir = cache_base_dir / cache_dir_name
    data_pairs = get_path_pairs(output_dir)
    seed = 42
    data_pairs = zip(*data_pairs)
    X_train, X_test, y_train, y_test = train_test_split(*data_pairs,
                                                        test_size=test_size, random_state=seed)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    train_dir = cache_dir / 'train'
    train_images_dir = train_dir / 'images'
    train_labels_dir = train_dir / 'labels'

    test_dir = cache_dir / 'test'
    test_images_dir = test_dir / 'images'
    test_labels_dir = test_dir / 'labels'

    for dir_ in (train_images_dir, train_labels_dir, \
                 test_images_dir, test_labels_dir):
        dir_.mkdir(exist_ok=True, parents=True)

    transfer_data(X_train, y_train, train_images_dir, train_labels_dir)
    transfer_data(X_test, y_test, test_images_dir, test_labels_dir)


def process(dataset_dir_path: str, cache_base_dir: str, max_side: int, df_name: str,
            source_img_dir: str, ds_dir_name: str, test_size: float):
    dataset_dir_path = Path(dataset_dir_path)
    segmentation_path = dataset_dir_path / ds_dir_name
    dataset_name = f'dataset_max_side_{max_side}'
    output_dir = segmentation_path / dataset_name
    output_images_path = output_dir / 'images'
    output_masks_path = output_dir / 'masks'

    output_images_path.mkdir(exist_ok=True, parents=True)
    output_masks_path.mkdir(exist_ok=True)

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = read_dataframe(df_path)

    images_path = dataset_dir_path / source_img_dir
    ds_size = len(df.index)

    for tag_id, row in tqdm(df.iterrows(), total=ds_size):
        rect = row['rect']
        if rect is None: continue
        image_path = get_file_sharding(images_path, tag_id)
        stem = Path(Path(image_path).stem)
        processed_img_path = output_images_path / stem.with_suffix(JPEG_SUFFIX)
        mask_path = output_masks_path / stem.with_suffix(NPY_SUFFIX)

        img = read_jpeg(image_path)

        # PROCESSING
        w, h, _ = img.shape
        mask = np.zeros((w, h), dtype=np.uint8)

        rect = read_json_str(rect)
        unirect = UniversalRect.from_coords_dict(rect)
        points = unirect.polygon.exterior.coords
        points = np.array([points], dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, WHITE_COLOR)

        img, mask = resize(img, mask, max_side)

        write_array(mask_path, mask)
        write_jpeg(processed_img_path, img)
    cache_base_dir = Path(cache_base_dir)
    prepare_for_train(output_dir, cache_base_dir, ds_dir_name, test_size)


def rm_non_exist_rows(dataset_dir_path: str, df_name: str, img_dir: str, save_df_name: str):
    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = read_dataframe(df_path)
    img_dir = dataset_dir_path / img_dir
    drop_ids = []
    for tag_id in df.index:
        img_path = get_file_sharding(img_dir, tag_id)
        if img_path.exists():
            continue
        drop_ids.append(tag_id)
    df.drop(index=drop_ids, inplace=True)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % save_df_name)
    write_dataframe(df, df_path)
