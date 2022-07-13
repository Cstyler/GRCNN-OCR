import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import ujson as json
from keras.preprocessing import image

from pylibs import jpeg_utils
from pylibs.storage_utils import get_file_sharding


@lru_cache()
def input_lens(max_text_len: int, batch_size: int) -> np.ndarray:
    a = np.ones((batch_size, 1)) * max_text_len
    return a.astype(np.float32, copy=False)


@lru_cache()
def zero_array(batch_size: int, dtype='float32') -> np.ndarray:
    return np.zeros(batch_size, dtype)


def add_none_symbol_to_labels(char_freqs: np.ndarray, max_text_len: int) -> Iterable[np.ndarray]:
    for char_freq in char_freqs:
        symbol_num = char_freq.sum()
        none_symbol_freq = max_text_len - symbol_num
        arr = np.append(char_freq, none_symbol_freq)
        yield arr


class BatchGenerator(image.Iterator):
    def __init__(self, df: pd.DataFrame,
                 batch_size: int, images_dir_path: Path, max_text_len: int,
                 transform_list: Optional[List[albu.BasicTransform]] = None,
                 augment_prob: float = 0.5,
                 shuffle: bool = True, seed: Optional[int] = None):
        self.df_index = df.index
        self.max_text_len = max_text_len
        char_freqs = df.char_freq.values
        char_freqs = np.stack(tuple(np.asarray(json.loads(x)) for x in char_freqs))
        char_freqs = add_none_symbol_to_labels(char_freqs, max_text_len)
        self.char_freqs = np.asarray(tuple(char_freqs))
        self.labels = np.stack([np.asarray(json.loads(x), np.float32) for x in df.label])
        self.label_lens = df.label_len.values.astype(np.float32, copy=False)
        self.texts = df.text.values
        self.images_dir_path = images_dir_path
        self.apply_augmentation = transform_list is not None
        if self.apply_augmentation:
            self.augmentations = albu.Compose(transform_list, p=augment_prob)
        n = len(self.df_index)
        super().__init__(n, batch_size, shuffle, seed)

    @property
    def steps_per_epoch(self):
        return math.ceil(self.n / self.batch_size)

    @staticmethod
    def normalize_img(img: np.ndarray) -> np.ndarray:
        img = 2 * (img / 255) - 1
        return img.astype(np.float32)

    def iter_images(self, df_index: pd.Index) -> Iterable[np.ndarray]:
        for tag_id in df_index:
            img_path = get_file_sharding(self.images_dir_path, tag_id)
            img = jpeg_utils.read_jpeg(str(img_path))
            img = self.augment_img(img)
            img = self.normalize_img(img)
            yield img

    def augment_img(self, img: np.ndarray) -> np.ndarray:
        if self.apply_augmentation:
            return self.augmentations(image=img)['image']
        return img

    def _get_batches_of_transformed_samples(self,
                                            index_array: np.ndarray) \
            -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Gets a batch of transformed samples.
        # Arguments
            index_array: array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        df_batch = self.df_index[index_array]
        inputs = np.stack(tuple(self.iter_images(df_batch)))
        labels = self.labels[index_array]
        texts = self.texts[index_array]
        batch_label_len = self.label_lens[index_array]
        char_freqs = self.char_freqs[index_array]
        batch_size = len(index_array)
        batch_input_len = input_lens(self.max_text_len, batch_size)

        batch_x = {
                "input"      : inputs,
                "labels"     : labels,
                "labels_freq": char_freqs,
                "input_len"  : batch_input_len,
                "label_len"  : batch_label_len,
                "texts"      : texts
        }

        batch_y = zero_array(batch_size)
        return batch_x, batch_y


def data_loader(image_dir: str) -> Iterable[np.ndarray]:
    image_dir = Path(image_dir)
    for p in image_dir.iterdir():
        img = jpeg_utils.read_jpeg(p)
        print(p.name)
        yield img


class InferenceBatchGenerator(image.Iterator):
    def __init__(self, df: pd.DataFrame,
                 batch_size: int, images_dir_path: Path, width: int, height: int, max_text_len: int):
        self.df_index = df.index
        self.labels = np.stack([np.asarray(json.loads(x), np.float32) for x in df.label])
        self.texts = df.text.values
        self.images_dir_path = images_dir_path
        self.label_lens = df.label_len.values.astype(np.float32, copy=False)
        self.resize_shape = (width, height)
        self.max_text_len = max_text_len
        n = len(self.df_index)
        super().__init__(n, batch_size, False, None)

    def iter_images(self, df_index: pd.Index) -> Iterable[np.ndarray]:
        for tag_id in df_index:
            img_path = get_file_sharding(self.images_dir_path, tag_id)
            img = jpeg_utils.read_jpeg(str(img_path))
            img = self.process_img(img)
            yield img

    def process_img(self, img: np.ndarray):
        img = cv2.resize(img, self.resize_shape)
        img = 2 * (img / 255) - 1
        img = img.astype(np.float32)
        return img

    def _get_batches_of_transformed_samples(self,
                                            index_array: np.ndarray) \
            -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        df_batch = self.df_index[index_array]
        inputs = np.stack(tuple(self.iter_images(df_batch)))
        labels = self.labels[index_array]
        texts = self.texts[index_array]
        batch_size = len(index_array)
        batch_input_len = input_lens(self.max_text_len, batch_size)
        batch_label_len = self.label_lens[index_array]
        batch_x = {
                "input"      : inputs,
                "labels"     : labels,
                "input_len"  : batch_input_len,
                "label_len"  : batch_label_len,
                "texts"      : texts
        }
        return batch_x
