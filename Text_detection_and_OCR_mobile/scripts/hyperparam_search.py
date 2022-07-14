import sys

sys.path.append("../..")
from tools.utils import allow_growth, enableGPU

gpu = 0
if gpu:
    gpu_id = 1
    enableGPU(gpu_id)
    allow_growth()

from hyperopt import hp

from price_detector.analysis.hyperparams import search_hyperparams_detection, \
    search_hyperparams_find_price


def main_det():
    model_name = 'digits_epoch-74_loss-0.1696_acc-0.9559.h5'
    # suffix = 'det_train'
    # boxes_arrays_filename = f'union_companies_labels_train_{suffix}'
    # img_list_filename = f'union_companies_imgs_train_{suffix}'
    suffix = 'det_train_2'
    boxes_arrays_filename = f'union_companies_labels_train_2_{suffix}'
    img_list_filename = f'union_companies_imgs_train_2_{suffix}'
    # save_suffix = "detection_v3"
    save_suffix = "detection_v4"
    max_queue_len = 3
    save_every_n_epoch = 5
    trials_file = None
    search_max_evals = 1000
    loss_threshold = -95.
    gt_thresh = .2
    space = [
        hp.uniform("region_box_area_ratio_threshold", .0001, .01),
        hp.uniform("same_iou_threshold", .6, .9),
        hp.uniform("iou_threshold", .6, .8),
        hp.uniform("area_ratio_threshold", .6, .8),
        hp.uniform("same_digit_iou_threshold", .5, .7),
        hp.uniform("small_box_ratio_threshold", .00005, .001),
        hp.uniform("min_ar", .1, .3),
        hp.uniform("max_ar", 1.2, 1.6),
        # hp.quniform("min_area", 4, 81, 2),
        hp.uniform("min_area", .00001, .001),
        hp.uniform("max_area", .2, .7),
        # hp.quniform("mser_min_area", 4, 81, 2),
        hp.uniform("mser_min_area", .00001, .001),
        hp.uniform("mser_max_area", .5, .7),
        hp.uniform("mser_max_variation", .2, .6),
        hp.quniform("mser_delta", 2, 7, 1),
        hp.quniform("box_expand_size", 1, 3, 1)
    ]
    result = search_hyperparams_detection(model_name, img_list_filename,
                                          boxes_arrays_filename, space,
                                          search_max_evals, loss_threshold,
                                          save_suffix,
                                          trials_file, gt_thresh,
                                          max_queue_len, save_every_n_epoch)
    print(result)


def main_fp():
    # suffix = "train"
    suffix = "train_2"
    img_list_filename = f"union_companies_imgs_{suffix}"
    prices_filename = f"union_companies_prices_{suffix}"
    # version = 'v9'
    version = 'v11'
    save_pickle = f"union_companies_labels_{suffix}-pred-boxes-{version}"

    # save_suffix = "fp_v11" # line segment distance v3(debugged), out37
    save_suffix = "fp_v12"  # new dataset, detection v11

    trials_file = None
    search_max_evals = 6000
    loss_threshold = -95.
    save_every_n_epoch = 200
    max_queue_len = 3
    space = [
        hp.quniform("thr_angle_price", 20, 50, 1),
        hp.quniform("thr_angle_rub", 20, 50, 1),
        hp.uniform("thr_distance_factor", .9, 3.),
        hp.uniform("thr_square_angle_diff", .01, 3.),
        hp.uniform("thr_dist_diff", .01, 3.),
        hp.uniform("coeffs_rub_angle", .01, 2.),
        hp.uniform("coeffs_rub_area", .01, 2.),
        hp.uniform("coeffs_rub_distance", .01, 2.),
        hp.uniform("coeffs_distance_x", .01, 2.),
        # hp.uniform("coeffs_price_distance", .01, 2.),
        # hp.uniform("coeffs_price_distance", 1.7, 1.8),
        hp.uniform("coeffs_price_distance", .001, 2.),
        hp.uniform("coeffs_price_area", .01, 2.),
    ]
    result = search_hyperparams_find_price(img_list_filename, prices_filename,
                                           version, space, search_max_evals, loss_threshold,
                                           save_pickle, save_suffix,
                                           trials_file, save_every_n_epoch, max_queue_len)
    print(result)


if __name__ == '__main__':
    main_fp()
    # main8_4()
    # main8_3()
    # main10()
