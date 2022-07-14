import operator
from collections import defaultdict
from pathlib import Path
from typing import Optional
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pylibs import img_utils, pandas_utils, pyplot_utils, text_utils
from pylibs.jpeg_utils import read_jpeg
from pylibs.numpy_utils import print_stats
from pylibs.pandas_utils import DF_FILE_FORMAT
from pylibs.storage_utils import get_file_sharding
from pylibs.types import FilterFunction
from .data_generation import InferenceBatchGenerator
from .model import GRCNN
from .predictor import ALPHABET, ProductCodeRecognizer
from .utils import calc_metric, find_model_path, set_gpu, text_to_labels


def test_nn(dataset_dir_path: str, df_name: str,
            model_num: int, epoch: int, batch_size: int,
            width: int, height: int, max_text_len: int,
            n_classes, grcl_niter, grcl_fsize, lstm_units,
            img_dir: str, full_img_dir: Optional[str], debug: bool = False,
            dist_filter: Optional[FilterFunction[float]] = None):
    gpu_id = 1080
    dataset_dir_path = Path(dataset_dir_path)
    images_dir_path = dataset_dir_path / img_dir
    full_img_dir = dataset_dir_path / full_img_dir

    test_set_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    test_df = pandas_utils.read_dataframe(test_set_path)
    # test_df = test_df.sample(frac=1).head(5000)
    test_gen = InferenceBatchGenerator(test_df, batch_size, images_dir_path, width, height, max_text_len)

    set_gpu(gpu_id)

    model_path = find_model_path(model_num, epoch)
    model = GRCNN(height, width, n_classes, max_text_len, grcl_fsize, grcl_niter, lstm_units)
    model.load_weights(model_path)

    _, predictions = model.predict_generator(test_gen)
    test_gen = InferenceBatchGenerator(test_df, 1, images_dir_path, width, height, max_text_len)

    dists = []
    dists_by_len = dict()
    rows = []
    for batch_dict, prediction, (tag_id, row) in zip(test_gen, predictions, test_df.iterrows()):
        true_text = row['text']
        if not true_text:
            continue
        true_text = "".join(x for x in true_text if x in ALPHABET)
        text_len = len(true_text)
        dist, pred_text = calc_metric(prediction, true_text, ALPHABET, False, debug, dist_filter, True)
        if debug and dist_filter(dist):
            rows.append((tag_id, pred_text, true_text, dist))
            print("Tag id:", tag_id)
            full_img = read_jpeg(get_file_sharding(full_img_dir, tag_id))
            img_utils.show_img(full_img, (3, 10))
            show_image((batch_dict['input'] + 1) / 2, (3, 10))
        true_num, len_count = dists_by_len.setdefault(text_len, (0, 0))
        dists_by_len[text_len] = true_num + (not dist), len_count + 1
        dists.append(dist)
    bad_samples_df = pd.DataFrame(rows, columns=('tag_id', 'pred', 'true', 'dist')).set_index('tag_id')
    pandas_utils.write_dataframe(bad_samples_df, dataset_dir_path / (DF_FILE_FORMAT % 'bad_samples'))

    acc_by_len = dict()
    for len_, (true_num, count) in dists_by_len.items():
        acc_by_len[len_] = round(true_num / count, 3)
    print(dists_by_len)
    print(acc_by_len)
    dists = np.asarray(dists)
    print_stats(dists)
    plt.hist(dists)
    plt.show()
    pyplot_utils.plot_hist_from_dict(acc_by_len)


def test_metric_by_markets(dataset_dir_path: str, df_name: str, img_dir: str,
                           model_num: int, epoch: int, batch_size: int, max_text_len: int,
                           height: int, width: int, n_classes: int,
                           grcl_niter: int, grcl_fsize: int, lstm_units: int,
                           accuracy_threshold: float):
    dataset_dir_path = Path(dataset_dir_path)
    images_dir_path = dataset_dir_path / img_dir

    test_set_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    test_df = pandas_utils.read_dataframe(test_set_path)
    test_gen = InferenceBatchGenerator(test_df, batch_size,
                                       images_dir_path, width, height, max_text_len)

    gpu_id = 1080
    set_gpu(gpu_id)
    rec = ProductCodeRecognizer(model_num, epoch, height, width,
                                n_classes, grcl_niter, grcl_fsize, lstm_units)
    pred_texts, pred_matrices = rec.recognize_batch(test_gen)
    dists = []
    true_labels = []
    true_lens = []
    dists_by_len = dict()
    dists_by_market = dict()
    market_key = 'market_id'
    text_key = 'text'
    total = len(test_df.index)
    for pred_text, (tag_id, row) in tqdm(zip(pred_texts, test_df.iterrows()), total=total, smoothing=.01):
        true_text = row[text_key]
        if not true_text:
            continue
        true_text = "".join(x for x in true_text if x in ALPHABET)
        labels = text_to_labels(true_text, max_text_len, ALPHABET)
        true_labels.append(labels)
        true_lens.append(len(labels))
        text_len = len(true_text)
        dist = text_utils.levenshtein_distance_weighted(pred_text, true_text, 1, 2, 1)
        true_num, total_num = dists_by_len.setdefault(text_len, (0, 0))
        true_predict = not dist
        dists_by_len[text_len] = true_num + true_predict, total_num + 1
        market = row[market_key]
        true_num, total_num = dists_by_market.setdefault(market, (0, 0))
        dists_by_market[market] = true_num + true_predict, total_num + 1
        dists.append(dist)
    ds_size = len(true_labels)
    costs = K.ctc_batch_cost(np.array(true_labels), pred_matrices,
                             np.ones((ds_size, 1)) * max_text_len, np.expand_dims(true_lens, 1))
    costs = K.squeeze(costs, 1)
    probs = K.get_value(K.exp(-costs))

    probs_by_market = defaultdict(list)
    for (_, row), dist, prob in zip(test_df.iterrows(), dists, probs):
        market = row[market_key]
        probs_by_market[market].append((dist, prob))

    thresholds = np.linspace(.0, np.max(probs), num=1000)
    skipped_percents = dict()
    for market, market_metrics in probs_by_market.items():
        metric_triples = []
        ds_size = len(market_metrics)
        for thr in thresholds:
            true_num = 0
            total = 0
            for dist, prob in market_metrics:
                thr_flag = prob > thr
                total += thr_flag
                dist_flag = thr_flag and dist == 0
                true_num += dist_flag
            acc = true_num / total if total else .0
            skipped_percent = total / ds_size
            triple = round(acc * 100, 3), round(thr, 6), round(skipped_percent * 100, 3)
            metric_triples.append(triple)
        filter_fun = lambda x: x[0] > accuracy_threshold and x[2] > 1.
        filtered_triples = list(filter(filter_fun, metric_triples))
        skipped_percents[market] = max(filtered_triples, key=operator.itemgetter(2))

    print(skipped_percents)
    acc_by_len = accuracy_dict_from_count_dict(dists_by_len)
    acc_by_market = accuracy_dict_from_count_dict(dists_by_market)
    print(dists_by_len)
    print(acc_by_len)
    print(dists_by_market)
    print(acc_by_market)
    dists = np.asarray(dists)
    print_stats(dists)
    print_stats(probs)
    plt.hist(dists)
    plt.show()
    plt.hist(np.round(probs, 7), bins=(.0001, .001, .002, .003, .004, .005, .01, .015, .02, .03))
    plt.show()
    pyplot_utils.plot_hist_from_dict(acc_by_len)


def show_image(batch_x, figsize=None):
    for x in batch_x:
        img_utils.show_img(x, figsize)


def test_google(dataset_dir_path: str, df_name: str):
    dataset_dir_path = Path(dataset_dir_path)
    dataset_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(dataset_path)
    dists = []
    dists_by_len = dict()
    default_tuple = (0, 0)

    for _, row in df.iterrows():
        true_text = row['text']
        if not true_text:
            continue
        pred_text = row['google_text']
        true_text = "".join(x for x in true_text if x in ALPHABET)
        text_len = len(true_text)
        dist = text_utils.levenshtein_distance_weighted(pred_text, true_text, 1, 2, 1)
        true_num, len_count = dists_by_len.setdefault(text_len, default_tuple)
        dists_by_len[text_len] = true_num + (not dist), len_count + 1
        dists.append(dist)

    acc_by_len = dict()
    for len_, (true_num, count) in dists_by_len.items():
        acc_by_len[len_] = round(true_num / count, 3)
    print(dists_by_len)
    print(acc_by_len)
    dists = np.asarray(dists)
    print_stats(dists)
    plt.hist(dists)
    plt.show()
    pyplot_utils.plot_hist_from_dict(acc_by_len)


def accuracy_dict_from_count_dict(count_dict: dict, round_ndigits=3) -> dict:
    acc_dict = dict()
    for k, (true_num, total_num) in count_dict.items():
        if total_num:
            acc_dict[k] = round(true_num / total_num, round_ndigits)
    return acc_dict
