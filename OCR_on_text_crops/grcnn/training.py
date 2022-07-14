import random
from typing import Optional, Tuple

import albumentations as albu
import numpy as np
import tensorflow as tf
from keras.optimizers import Optimizer

from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe
from .callbacks import get_callbacks
from .data_generation import BatchGenerator
from .model import Model
from .utils import DATASET_DIR, find_model_path, set_gpu

seed = 42
gpu_id = 1080

np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)


def loss_fun(y_true, y_pred):
    return y_pred


def compile_model(model: Model, optimizer: Optimizer, loss_key: str):
    loss_dict = dict(predictions=None)
    loss_dict[loss_key] = loss_fun
    model.compile(loss=loss_dict, optimizer=optimizer)


def get_transforms():
    # min_height = int(.88 * height)
    # max_height = int(.96 * height)
    # w2h_ratio = width / height
    WHITE_PIXEL = 255
    # ps = [.4, .3, .9]
    # ps = [.6, .6, .9]
    ps = [.5, .5]
    transform_list = [
            albu.OneOf(
                    [
                            # pixel augmentations
                            albu.IAAAdditiveGaussianNoise(loc=0,
                                                          scale=(.01 * WHITE_PIXEL, .05 * WHITE_PIXEL), p=1),
                            albu.RandomBrightnessContrast(contrast_limit=.2,
                                                          brightness_limit=.2, p=1),
                            albu.RandomGamma(gamma_limit=(80, 120), p=1),
                    ],
                    p=ps[0],
            ),
            albu.OneOf(
                    [
                            # convolutional augmentations
                            albu.IAASharpen(alpha=(.05, .2), lightness=(.91, 1.), p=1),
                            albu.Blur(blur_limit=3, p=1),
                            albu.MotionBlur(blur_limit=3, p=1),
                            albu.MedianBlur(blur_limit=3, p=1),
                    ],
                    p=ps[1],
            ),
            # albu.ShiftScaleRotate(shift_limit=.02, scale_limit=.02,
            #                       rotate_limit=2, border_mode=cv2.BORDER_REFLECT_101,
            #                       interpolation=cv2.INTER_LINEAR, p=ps[2])
    ]
    return transform_list


def train(train_df_name: str, val_df_name: str, img_dir: str,
          model_num: int, initial_epoch: int, epochs: int,
          train_augment_prob: float, batch_size: int,
          pretrain_model_path: str, optimizer: Optimizer,
          width: int, height: int, max_text_len: int, n_classes: int,
          grcl_niter: int, grcl_fsize: int, lstm_units: int,
          loss_weights: Optional[Tuple[float, float]] = None, lookahead_optimizer: Optional = None):
    transform_list = get_transforms()
    images_dir_path = DATASET_DIR / img_dir

    train_set_path = DATASET_DIR / (DF_FILE_FORMAT % train_df_name)
    train_df = read_dataframe(train_set_path)
    train_gen = BatchGenerator(train_df, batch_size, images_dir_path, max_text_len,
                               transform_list, train_augment_prob, seed=seed)

    val_set_path = DATASET_DIR / (DF_FILE_FORMAT % val_df_name)
    val_df = read_dataframe(val_set_path)
    val_gen = BatchGenerator(val_df, batch_size, images_dir_path, max_text_len, shuffle=False)

    set_gpu(gpu_id)

    callbacks = get_callbacks(model_num, train_gen, val_gen)

    # TODO change this to model config : dict str->value(float, sequence[float], etc.)
    if loss_weights:
        from .model import GRCNN_mse
        loss_key = 'total_loss'
        model = GRCNN_mse(height, width, n_classes, max_text_len, grcl_fsize, grcl_niter, lstm_units, loss_weights)
    else:
        from .model import GRCNN
        loss_key = 'ctc'
        model = GRCNN(height, width, n_classes, max_text_len, grcl_fsize, grcl_niter, lstm_units)

    if initial_epoch:
        model_file_path = find_model_path(model_num, initial_epoch)
        print(f"Model found: {model_file_path}")
        model.load_weights(model_file_path)
    if pretrain_model_path:
        model.load_weights(pretrain_model_path)
    compile_model(model, optimizer, loss_key)
    if lookahead_optimizer:
        lookahead_optimizer.inject(model)

    model.fit_generator(train_gen, steps_per_epoch=train_gen.steps_per_epoch, epochs=epochs,
                        verbose=1, callbacks=callbacks, use_multiprocessing=True,
                        workers=3, max_queue_size=20, initial_epoch=initial_epoch,
                        validation_data=val_gen, validation_steps=val_gen.steps_per_epoch)
