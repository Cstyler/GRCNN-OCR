import functools
import itertools
from typing import Callable, List, Optional

import numpy as np
from keras import backend as K
from keras.callbacks import (CSVLogger, Callback, EarlyStopping,
                             ModelCheckpoint, ReduceLROnPlateau, TensorBoard)

from .data_generation import BatchGenerator
from .predictor import ALPHABET
from .utils import MODELS_DIR, calc_metric, TENSORBOARD_DIR, LOGS_DIR


class MetricCallback(Callback):
    def __init__(self, generator: BatchGenerator,
                 metric_function: Callable,
                 log_name: str, true_label_key: str, batches_num: Optional[int] = None,
                 round_ndigits: int = 4):
        super().__init__()
        self.gen = generator
        self.metric_fun = metric_function
        self.log_name = log_name
        self.true_label_key = true_label_key
        self.batches_num = batches_num
        self.round_ndigits = round_ndigits

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.gen.reset()
        metrics = self.calculate_target()
        self.gen.reset()
        leven = np.mean(metrics)
        acc = np.sum(metrics == 0) / len(metrics)
        logs[self.log_name + '_leven'] = round(leven, self.round_ndigits)
        logs[self.log_name + '_acc'] = round(acc * 100, 2)

    def calculate_target(self) -> np.ndarray:
        metrics = []
        for batch_x, _ in itertools.islice(self.gen, self.batches_num):
            _, y_pred = self.model.predict_on_batch(batch_x)
            for y_true, y_pred in zip(batch_x[self.true_label_key], y_pred):
                score = self.metric_fun(y_pred, y_true)
                metrics.append(score)
        metrics = np.array(metrics)
        return metrics


class CustomTensorBoard(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update(lr=K.eval(self.model.optimizer.lr))
        super().on_epoch_end(epoch, logs)


def get_callbacks(model_num: int, train_gen: BatchGenerator, val_gen: BatchGenerator) -> List[Callback]:
    csv_filename = LOGS_DIR / f'model{model_num}.csv'
    csvLoggerCallback = CSVLogger(csv_filename, append=True)
    tensorboard_folder = TENSORBOARD_DIR / f'run{model_num}'
    tensorboard_folder.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = CustomTensorBoard(str(tensorboard_folder), write_graph=True,
                                             batch_size=train_gen.batch_size,
                                             write_grads=True, write_images=True)
    metric_function = functools.partial(calc_metric, alphabet=ALPHABET)
    valMetricCallback = MetricCallback(val_gen, metric_function,
                                       'val', 'texts', val_gen.steps_per_epoch)
    trainMetricCallback = MetricCallback(train_gen, metric_function, 'train', 'texts', 10)
    # batches_num = 100
    # ctc_norm_callback = NormGradients(train_gen, 'ctc', batches_num)
    # freq_norm_callback = NormGradients(train_gen, 'mse', batches_num)

    earlyStoppingCallback = EarlyStopping('loss', patience=5, verbose=1)
    # reduceLRCallback = ReduceLROnPlateau(monitor='val_acc', factor=.5, mode='max',
    #                                      patience=5, verbose=1,
    #                                      min_lr=1e-8, min_delta=0)
    reduceLRCallback = ReduceLROnPlateau('val_loss', patience=10, factor=.5,
                                         verbose=1, min_lr=1e-8, min_delta=0)
    model_dir = MODELS_DIR / f'model{model_num}'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file_path = model_dir / 'epoch_{epoch}_val_leven{val_leven:.6f}.hdf5'
    modelCheckpointCallback = ModelCheckpoint(str(model_file_path), monitor='val_leven',
                                              verbose=0, save_best_only=True)
    callbacks = [valMetricCallback, trainMetricCallback,
                 modelCheckpointCallback, csvLoggerCallback,
                 earlyStoppingCallback, tensorboard_callback, reduceLRCallback]
    return callbacks
