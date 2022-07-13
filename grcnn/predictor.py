from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from keras.layers import (Activation, BatchNormalization, Bidirectional, Conv2D, Input, LSTM, MaxPool2D, ReLU, Reshape,
                          Softmax, ZeroPadding2D)
from keras.layers.merge import add, multiply
from keras.models import Model

from pylibs import minio_utils
from pylibs.numpy_utils import top_k
from .utils import decode_label, find_model_path

ALPHABET = "".join(map(str, range(10)))
# ALPHABET = "".join(map(str, range(10))) + '-'
max_text_len = 26
BATCH_SIZE = 8


def normalize_img(img: np.ndarray) -> np.ndarray:
    img = 2 * (img / 255) - 1
    return img.astype(np.float32)


relu = ReLU()


def GRCL(inp, n_out: int, n_iter: int, f_size: int):
    padding = 'same'

    conv_rec = Conv2D(n_out, 3, padding=padding)  # shared weights
    conv_gate_rec = Conv2D(n_out, 1)  # shared weights

    # Gated
    if f_size == 1:
        y = Conv2D(n_out, f_size)(inp)
    else:
        y = Conv2D(n_out, f_size, padding=padding)(inp)

    bn_gate_f = BatchNormalization()(y)

    # Feed forward
    y = Conv2D(n_out, 3, padding=padding)(inp)
    bn_f = BatchNormalization()(y)

    x = relu(bn_f)
    for _ in range(n_iter - 1):
        y = conv_rec(x)
        bn_rec = BatchNormalization()(y)

        y = conv_gate_rec(x)
        y = BatchNormalization()(y)
        y = add([y, bn_gate_f])
        gate = Activation('sigmoid')(y)

        y = multiply([bn_rec, gate])
        y = BatchNormalization()(y)
        y = add([bn_f, y])

        x = relu(y)
    return x


def GRCNN(height: int, width: int, n_classes: int, grcl_fsize: int, grcl_niter: int, lstm_units: int) -> Model:
    pool_size = 2
    input_layer = Input(name="input", shape=(height, width, 3))
    x = Conv2D(64, 3, padding='same')(input_layer)
    x = BatchNormalization()(x)  # TODO it was not here
    x = relu(x)
    x = MaxPool2D(pool_size=pool_size, strides=2)(x)
    x = GRCL(x, 64, grcl_niter, grcl_fsize)
    x = MaxPool2D(pool_size=pool_size, strides=2)(x)
    x = GRCL(x, 128, grcl_niter, grcl_fsize)
    x = ZeroPadding2D(padding=(0, 1))(x)
    strides = (2, 1)
    x = MaxPool2D(pool_size=pool_size, strides=strides)(x)
    x = GRCL(x, 256, grcl_niter, grcl_fsize)
    x = ZeroPadding2D(padding=(0, 1))(x)
    strides = (2, 1)  # in orig (2, 1)
    x = MaxPool2D(pool_size=pool_size, strides=strides)(x)
    x = Conv2D(512, 2)(x)  # no padding
    x = BatchNormalization()(x)
    x = relu(x)

    w = int(x.shape[1] * x.shape[2])  # in orig x.shape[1] == 1
    h = int(x.shape[3])
    x = Reshape((w, h))(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='sum')(x)
    units = n_classes + 1
    x = Bidirectional(LSTM(units, return_sequences=True), merge_mode='sum')(x)
    y_pred = Softmax(name='predictions')(x)
    model = Model(inputs=input_layer, outputs=y_pred)
    return model


class ProductCodeRecognizer:
    def __init__(self, model_num: int, epoch: int,
                 height, width, n_classes, grcl_niter, grcl_fsize, lstm_units, debug=False):
        model_path = find_model_path(model_num, epoch)
        self.model = GRCNN(height, width, n_classes, grcl_niter, grcl_fsize, lstm_units)
        self.model.load_weights(model_path)
        self.height = height
        self.width = width
        self.debug = debug

    def recognize_batch(self, images: Iterable[dict]) -> Tuple[Iterable[str], np.ndarray]:
        predictions = self.model.predict_generator(images)
        return map(self.get_text_from_nn_output, predictions), predictions

    def recognize(self, image: np.ndarray) -> str:
        x = np.expand_dims(normalize_img(image), -1)
        pred = self.model.predict(x, 1)
        return self.get_text_from_nn_output(pred)

    def get_text_from_nn_output(self, pred: np.ndarray) -> str:
        pred_argmax = pred.argmax(axis=1)
        pred_text = decode_label(pred_argmax, ALPHABET)
        if self.debug:
            print("Pred argmax", "".join('•' if x == len(ALPHABET) else str(x) for x in pred_argmax))
            for i, x in enumerate(pred):
                ind = top_k(x, 3)
                print(f"it{i}. Pred", ind, np.round(x[ind], 3))
        return pred_text

    @staticmethod
    def download_from_minio(s3_path: str, save_dir: str, model_filename: str) -> str:
        save_path = Path(save_dir) / model_filename
        if not save_path.exists() or not minio_utils.are_equal(save_path, s3_path, minio_utils.DEFAULT_MINIO_CONFIG):
            print('%s. Start download' % s3_path)
            minio_utils.download_from_minio(s3_path, save_path, minio_utils.DEFAULT_MINIO_CONFIG)
            print('%s. Downloaded' % s3_path)
        else:
            print('%s. No need for downloading' % s3_path)
        return str(save_path)
