import os
from collections import deque
from pathlib import Path
from typing import Optional, List

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model, Model

from pylibs import text_utils
from pylibs.types import FilterFunction
import pandas as pd
from grcnn.model import GRCNN

PROJECT_NAME = 'product_code_ocr'
LOGS_DIR = Path(f"/opt/agazade/data_science/training/checkpoints/{PROJECT_NAME}/logs")
MODELS_DIR = Path(f'/srv/data_science/training/checkpoints/{PROJECT_NAME}')
TENSORBOARD_DIR = Path(f'/srv/data_science/tensorboard/training/{PROJECT_NAME}')
DATASET_DIR = Path(f'/srv/data_science/storage/{PROJECT_NAME}')
for d in (MODELS_DIR, LOGS_DIR, TENSORBOARD_DIR, DATASET_DIR):
    d.mkdir(exist_ok=True, parents=True)


def find_model_path(model_num: int, epoch: int) -> str:
    model_dir = MODELS_DIR / f'model{model_num}'
    try:
        path, *_ = model_dir.glob(f'epoch_{epoch}*.hdf5')
        return str(path)
    except ValueError:
        raise ValueError(f"Model not found. Epoch: {epoch}. Model dir: {model_dir}")


def find_best_model_epoch(model_num: int) -> int:
    csv_path = LOGS_DIR / f'model{model_num}.csv'
    df = pd.read_csv(csv_path)
    return int(df.val_leven.idxmin()) + 1


def load_model_grcnn(model_num, epoch, height, width, n_classes, max_text_len, grcl_fsize, grcl_niter,
                     lstm_units) -> Model:
    model = GRCNN(height, width, n_classes, max_text_len, grcl_fsize, grcl_niter, lstm_units)
    path = find_model_path(model_num, epoch)
    model.load_weights(path)
    return model


def find_and_load_model(model_num: int, epoch: int,
                        custom_objects=None, compile_: bool = False,
                        debug: bool = False) -> Model:
    model_file_path = find_model_path(model_num, epoch)
    model = load_model(model_file_path, custom_objects, compile_)
    if debug:
        model.summary()
    return model


def decode_label(prediction: np.ndarray, alphabet: str) -> str:
    """
    decodes one-hot representation of text (from network or from OneHotEncoder)
    :param predictions: N x A matrix, where N is batch size, and A is alphabet capacity + 1
    :return: two arrays with decoded strings: with double symbols and "blanks" and without
    """
    extra_alphabet = alphabet + '.'
    alphabet_size = len(alphabet)
    stack = deque(maxlen=len(prediction))
    peeled_stack = deque(maxlen=len(prediction))
    for i, char_ord in enumerate(prediction[:-1]):
        stack.append(extra_alphabet[int(char_ord)])
        if char_ord != alphabet_size and prediction[i + 1] != char_ord:
            peeled_stack.append(alphabet[int(char_ord)])
    last_ord = prediction[-1]

    stack.append(extra_alphabet[int(last_ord)])
    if last_ord != alphabet_size:
        peeled_stack.append(alphabet[int(last_ord)])
    return "".join(peeled_stack)


def calc_metric(pred: np.ndarray,
                true_text: str,
                alphabet: str,
                filter_text: bool = True, debug: bool = False,
                dist_filter: Optional[FilterFunction[float]] = None,
                return_pred_text: bool = False) -> float:
    pred_argmax = pred.argmax(axis=1)
    pred_text = decode_label(pred_argmax, alphabet)
    if filter_text:
        true_text = "".join(x for x in true_text if x in alphabet)

    dist = text_utils.levenshtein_distance_weighted(pred_text, true_text, 1, 2, 1)
    if debug and dist_filter and dist_filter(dist):
        print("True text:", true_text, ", Len:", len(true_text), ", Pred text:", pred_text,
              ", Dist:", dist)
        print("Pred argmax", "".join('•' if x == len(alphabet) else str(x) for x in pred_argmax))
        from pylibs.numpy_utils import top_k
        for i, x in enumerate(pred):
            ind = top_k(x, 3)
            print(f"sym{i}. Pred", ind, np.round(x[ind], 3))
    return (dist, pred_text) if return_pred_text else dist


def set_gpu(gpu_id: int, gpu_mem_fraction: float = 0.8):
    devices_dict = {1080: "0"}
    gpu_num = devices_dict[gpu_id]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
    config.log_device_placement = True
    tf.logging.set_verbosity(tf.logging.ERROR)
    sess = tf.Session(config=config)
    K.set_session(sess)


def text_to_labels(text: str, max_len: int, alphabet: List[str], filter_text: bool = True) -> List[int]:
    char_set = set(alphabet)
    n_classes = len(alphabet)
    chars2num = {x: i for i, x in enumerate(alphabet)}

    text_list = list(filter(char_set.__contains__, text) if filter_text else text)
    len_vals = len(text_list)
    extend_size = max_len - len_vals
    feature_list = list(map(chars2num.__getitem__, text_list)) + [n_classes] * extend_size
    return feature_list
