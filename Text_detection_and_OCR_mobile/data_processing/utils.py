import json
import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from price_detector.data_processing.numpy_utils import NPY_SUFFIX, read_array, write_array
from price_detector.data_processing.pandas_utils import DF_FILE_FORMAT, read_dataframe, \
    write_dataframe

MAIN_DIR = Path('..')
DATASETS_DIR = MAIN_DIR / 'datasets'
BBOX_KEYS = ("class", "xmin", "ymin", "xmax", "ymax")


def read_array_local(filename: str) -> np.ndarray:
    return read_array(DATASETS_DIR / (filename + NPY_SUFFIX))


def write_array_local(filename: str, arr: np.ndarray):
    return write_array(DATASETS_DIR / (filename + NPY_SUFFIX), arr)


def write_json_local(obj, filename, encoding='utf-8'):
    json_path = DATASETS_DIR / f"{filename}.json"
    write_json(json_path, obj, encoding)


def write_json(json_path, obj, encoding='utf-8'):
    with open(str(json_path), 'w', encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=False)


def read_json_local(filename: str):
    json_path = DATASETS_DIR / f"{filename}.json"
    return read_json(json_path)


def read_json(json_path):
    with open(str(json_path), 'r') as f:
        return json.load(f)


def annotation_list_to_array(anno_list: List[Dict[str, Union[str, int]]],
                             bbox_keys=BBOX_KEYS):
    return np.asarray([[bbox[key] for key in bbox_keys] for bbox in anno_list])


def read_pickle_local(filename: str):
    path = DATASETS_DIR / (filename + ".pickle")
    return read_pickle(path)


def read_pickle(path):
    with open(str(path), 'rb') as file_:
        return pickle.load(file_)


def read_df(df_name):
    df_path = DATASETS_DIR / (DF_FILE_FORMAT % df_name)
    return read_dataframe(df_path)


def write_df(df_name, df):
    df_path = DATASETS_DIR / (DF_FILE_FORMAT % df_name)
    return write_dataframe(df, df_path)


def write_pickle_local(filename: str, obj):
    pickle_path = DATASETS_DIR / (filename + ".pickle")
    write_pickle(obj, pickle_path)


def write_pickle(obj, pickle_path):
    with open(str(pickle_path), 'wb') as file_:
        pickle.dump(obj, file_)


def digits_to_number(nums: List[int]):
    tot = 0
    for num in nums:
        tot *= 10
        tot += num
    return tot