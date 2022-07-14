from pathlib import Path
from typing import Union

import numpy as np

NPY_SUFFIX = '.npy'
Pathlike = Union[Path, str]


def write_array(path: Pathlike, array: np.ndarray):
    if not isinstance(path, str):
        path = str(path)
    np.save(path, array)


def read_array(path: Pathlike):
    if not isinstance(path, str):
        path = str(path)
    return np.load(path)


def print_stats(a: np.ndarray):
    np_mean = np.mean(a)
    sigma3 = 2 * np.std(a)
    print("Mean:", prettify_float(np_mean), ", Median:",
          prettify_float(np.median(a)),
          "3-sigma: (%s, %s)" % (
              prettify_float(np_mean - sigma3), prettify_float(np_mean + sigma3)),
          ", Min:",
          prettify_float(np.min(a)), ", Max:", prettify_float(np.max(a)))


def prettify_float(min_area, n_digits=5):
    return round(float(min_area), n_digits)
