from typing import Tuple

from keras import backend as K
from keras.layers import (Activation, BatchNormalization, Bidirectional, Conv2D, CuDNNLSTM, Input, Lambda, MaxPool2D,
                          ReLU, Reshape, Softmax, ZeroPadding2D)
from keras.layers.merge import add, multiply
from keras.losses import mean_squared_error
from keras.models import Model


def ctc_lambda_func(args: tuple):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def mse_func(args: tuple):
    y_true, y_pred = args
    return mean_squared_error(y_true, y_pred)


def calc_frequency(x):
    return K.sum(x, axis=1)


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


def GRCNN(height: int, width: int, n_classes: int,
          max_text_len: int, grcl_fsize: int, grcl_niter: int, lstm_units: int) -> Model:
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
    x = Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True), merge_mode='sum')(x)
    units = n_classes + 1
    x = Bidirectional(CuDNNLSTM(units, return_sequences=True), merge_mode='sum')(x)
    y_pred = Softmax(name='predictions')(x)

    dtype = 'float32'
    labels = Input(name='labels', shape=[max_text_len], dtype=dtype)
    SCALAR_SHAPE = [1]
    input_length = Input(name='input_len', shape=SCALAR_SHAPE, dtype=dtype)
    label_length = Input(name='label_len', shape=SCALAR_SHAPE, dtype=dtype)

    loss_layer = Lambda(ctc_lambda_func, name='ctc')
    ctc_loss = loss_layer([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_layer, labels, input_length, label_length], outputs=[ctc_loss, y_pred])
    return model


def GRCNN_mse(height: int, width: int, n_classes: int,
              max_text_len: int, grcl_fsize: int,
              grcl_niter: int, lstm_units: int,
              loss_weights: Tuple[float, float]) -> Model:
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
    x = Bidirectional(CuDNNLSTM(lstm_units, return_sequences=True), merge_mode='sum')(x)
    units = n_classes + 1
    x = Bidirectional(CuDNNLSTM(units, return_sequences=True), merge_mode='sum')(x)
    y_pred = Softmax(name='predictions')(x)

    dtype = 'float32'
    labels = Input(name='labels', shape=[max_text_len], dtype=dtype)
    labels_freq = Input(name='labels_freq', shape=[n_classes + 1], dtype=dtype)
    SCALAR_SHAPE = [1]
    input_length = Input(name='input_len', shape=SCALAR_SHAPE, dtype=dtype)
    label_length = Input(name='label_len', shape=SCALAR_SHAPE, dtype=dtype)

    loss_layer = Lambda(ctc_lambda_func, name='ctc')
    ctc_loss = loss_layer([y_pred, labels, input_length, label_length])
    pred_freq = Lambda(calc_frequency, name='pred_freq')(y_pred)
    mse_loss = Lambda(mse_func, name='mse')([labels_freq, pred_freq])
    sum_losses_lambda = lambda losses: sum(loss_weights[i] * x for i, x in enumerate(losses))
    total_loss = Lambda(sum_losses_lambda, SCALAR_SHAPE, name='total_loss')([ctc_loss, mse_loss])

    inputs = [input_layer, labels, input_length, label_length, labels_freq]
    model = Model(inputs=inputs, outputs=[total_loss, y_pred])
    return model
