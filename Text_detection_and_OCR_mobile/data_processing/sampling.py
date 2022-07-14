from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_processing import DATASETS_DIR, DF_FILE_FORMAT, read_array_local, \
    write_array_local, write_dataframe
from .pandas_utils import read_dataframe

SEED = 42

np.random.seed(SEED)


def train_test_split_array(imgs_filename: str, labels_filename: str,
                           test_size: float, suffix1='_train', suffix2='_test'):
    imgs = read_array_local(imgs_filename)
    labels = read_array_local(labels_filename)
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(imgs, labels,
                                                                        test_size=test_size)
    print(len(labels_test), len(labels_train))
    write_array_local(imgs_filename + suffix1, imgs_train)
    write_array_local(labels_filename + suffix1, labels_train)
    write_array_local(imgs_filename + suffix2, imgs_test)
    write_array_local(labels_filename + suffix2, labels_test)


def train_val_test_split(df: pd.DataFrame, val_size: float, test_size: float) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_len = len(df.index)
    val_index = int(df_len * (1 - (val_size + test_size)))
    test_index = int(df_len * (1 - test_size))
    shuffled_df = df.sample(frac=1)
    return np.split(shuffled_df, (val_index, test_index))


def train_val_split(df: pd.DataFrame, train_size: float) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled_df = df.sample(frac=1)
    return np.split(shuffled_df, [train_size])


def split_dataset(split_df_name: str, train_size: int,
                  df_name1: str, df_name2: str):
    df_path = DATASETS_DIR / (DF_FILE_FORMAT % split_df_name)
    df = read_dataframe(df_path)
    print("DF size:", len(df.index))
    x1, x2 = train_val_split(df, train_size)
    set_path1 = DATASETS_DIR / (DF_FILE_FORMAT % df_name1)
    print("size1:", len(x1.index))
    write_dataframe(x1, set_path1, index=True)
    set_path2 = DATASETS_DIR / (DF_FILE_FORMAT % df_name2)
    print("size2:", len(x2.index))
    write_dataframe(x2, set_path2, index=True)
