import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from SkynetCV import SkynetCV
import cv2
import numpy as np
from tqdm import tqdm

from bounding_box_utils.bounding_box_utils import iou
from price_detector.blur.blur import calc_mean_blur_metrics
from price_detector.detector.utils import show_boxes_on_image
from .numpy_utils import read_array
from .pandas_utils import DF_FILE_FORMAT, pd, write_dataframe
from .utils import annotation_list_to_array, digits_to_number, read_array_local, read_df, \
    read_pickle_local, \
    write_array_local, write_pickle_local
from .xml_processing import parse_xml

INDEX_NAME = 'tag_id'
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
MAIN_DIR = Path('..')
MODELS_DIR = MAIN_DIR / 'models'
DATASETS_DIR = MAIN_DIR / 'datasets'
BBOX_KEYS = ("class", "xmin", "ymin", "xmax", "ymax")
PNG_FORMAT = "%s.png"


def create_df(dataset_name: str, save_df_name: str):
    dataset_path = DATASETS_DIR / dataset_name
    annotations_path = dataset_path / 'annotations'
    rows = []
    failed_tags = 0
    for ann_path in tqdm(annotations_path.iterdir()):
        tag_id, boxes, image_shape = parse_xml(ann_path)
        if boxes is None:
            failed_tags += 1
            continue
        img_h, img_w = image_shape
        rows.append((tag_id, boxes, img_h, img_w))
    print("Failed tags:", failed_tags)
    df = pd.DataFrame(rows, columns=("tag_id", "boxes", "img_h", "img_w"))
    df.set_index(INDEX_NAME, inplace=True)

    save_df_name = DF_FILE_FORMAT % save_df_name
    save_df_path = DATASETS_DIR / save_df_name
    write_dataframe(df, save_df_path)


def merge_dfs(df_name1: str, df_name2: str, new_df_name: str):
    df1 = read_df(df_name1)
    df2 = read_df(df_name2)
    new_df = df1.append(df2).sample(frac=1)
    new_path = DATASETS_DIR / (DF_FILE_FORMAT % new_df_name)
    write_dataframe(new_df, new_path)


def merge_arrays(companies: List[str],
                 dataset_prefix: str,
                 dataset_suffix: str):
    res_imgs, res_labels = [], []
    for comp in companies:
        imgs_filename = f'{comp}_imgs_{dataset_suffix}'
        labels_filename = f'{comp}_annotations_{dataset_suffix}'
        imgs = read_array_local(imgs_filename)
        res_imgs.append(imgs)
        labels = read_array_local(labels_filename)
        res_labels.append(labels)
    res_imgs = np.concatenate(res_imgs)
    res_labels = np.concatenate(res_labels)
    print(f"Len imgs: {len(res_imgs)}")
    save_name = f'{dataset_prefix}_imgs_{dataset_suffix}'
    write_array_local(save_name, res_imgs)
    save_name2 = f'{dataset_prefix}_labels_{dataset_suffix}'
    write_array_local(save_name2, res_labels)
    print(save_name, save_name2)


def create_img_lists(images_dir: str,
                     img_list_filename: str,
                     max_side: int = 120):
    images_path = DATASETS_DIR / images_dir

    img_list = []
    for img_path in images_path.iterdir():
        img = cv2.imread(str(img_path))
        img_h, img_w, _ = img.shape
        if img_h > img_w:
            h = max_side
            scale_y = h / img_h
            w = int(scale_y * img_w)
        else:
            w = max_side
            scale_x = w / img_w
            h = int(scale_x * img_h)
        img = SkynetCV.resize(img, w, h)
        img_list.append(img)
    write_pickle_local(img_list_filename, img_list)


def create_img_boxes_lists(df_name: str, images_dir: str, arrays_filename: str,
                           img_list_filename: str, tag_ids_filename: str,
                           max_side: int = 120):
    df = read_df(df_name)
    images_path = DATASETS_DIR / images_dir
    resize_flag = max_side > 0

    arrays_list, array_inds, img_list, tag_ids = [], [], [], []
    for ind, (tag_id, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        tag_ids.append(tag_id)
        img_path = images_path / (PNG_FORMAT % tag_id)
        img = cv2.imread(str(img_path))
        if resize_flag:
            img_h, img_w, _ = img.shape
            if img_h > img_w:
                h = max_side
                scale_y = h / img_h
                w = int(scale_y * img_w)
                scale_x = w / img_w
            else:
                w = max_side
                scale_x = w / img_w
                h = int(scale_x * img_h)
                scale_y = h / img_h
            img = SkynetCV.resize(img, w, h)
        img_list.append(img)

        ann_arr = annotation_list_to_array(row.boxes).astype(np.float32)
        if resize_flag:
            ann_arr[:, [2, 4]] *= scale_y
            ann_arr[:, [1, 3]] *= scale_x
        ann_arr[:, 0] -= 1
        ann_array = ann_arr.astype(np.uint16, copy=False)
        arrays_list.append(ann_array)
    tag_ids = np.array(tag_ids)
    write_pickle_local(tag_ids_filename, tag_ids)
    write_pickle_local(arrays_filename, arrays_list)
    write_pickle_local(img_list_filename, img_list)


def boxes_features(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    xmaxs = boxes[:, 2]
    xmins = boxes[:, 0]
    ymins = boxes[:, 1]
    ymaxs = boxes[:, 3]
    widths = (xmaxs - xmins) / w
    heights = (ymaxs - ymins) / h
    x_centers = (xmaxs * 3 - xmins) * .5
    x_centers_norm = x_centers / w
    y_centers = (ymaxs * 3 - ymins) * .5
    y_centers_norm = y_centers / h
    x_dists = np.abs(x_centers_norm[:, np.newaxis] - x_centers_norm)
    y_dists = np.abs(y_centers_norm[:, np.newaxis] - y_centers_norm)

    features = [np.mean(widths), np.median(widths), np.std(widths), np.max(widths),
                np.mean(heights), np.median(heights), np.std(heights), np.max(heights),
                np.mean(x_centers_norm), np.median(x_centers_norm),
                np.std(x_centers_norm), np.max(x_centers_norm),
                np.mean(y_centers_norm), np.median(y_centers_norm),
                np.std(y_centers_norm), np.max(y_centers_norm),
                np.mean(x_dists), np.median(x_dists), np.std(x_dists), np.max(x_dists),
                np.mean(y_dists), np.median(y_dists), np.std(y_dists), np.max(y_dists)]
    # len(boxes)]
    return np.array(features)


MAX_NUM_BOXES = 15

MIN_NUM_BOXES = 1


def create_box_features(df_name: str, save_name: str,
                        verbose: bool = False, images_dir: Optional[str] = None):
    df = read_df(df_name)
    if images_dir is not None:
        images_path = DATASETS_DIR / images_dir
    bbox_keys = BBOX_KEYS[1:]
    res = defaultdict(list)
    for tag_id, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
        bboxes = annotation_list_to_array(row.boxes, bbox_keys)
        features = boxes_features(bboxes, row.img_w, row.img_h)
        res[len(bboxes)].append(features)
        if images_dir is not None:
            img_path = images_path / (PNG_FORMAT % tag_id)
            img = cv2.imread(str(img_path))
            show_boxes_on_image(bboxes, img)
    write_pickle_local(save_name, res)


def save_box_cluster_members_pred(images_dir: str,
                                  clusters_array_name: str, n_members: int,
                                  arrays_filename: str,
                                  img_list_filename: str):
    clusters = read_array_local(clusters_array_name)
    boxes_list = []
    images_path = DATASETS_DIR / images_dir

    img_list = []
    for img_path in images_path.iterdir():
        img = cv2.imread(str(img_path))
        # pred = detect_digits(...)
        img_list.append(img)
    res_img_list = []
    res_boxes_list = []
    n_clusters = len(np.unique(clusters))
    for cluster in range(n_clusters):
        cluster_members = np.where(clusters == cluster)[0]
        img_indices = np.random.choice(cluster_members, n_members)
        for i in img_indices:
            res_boxes_list.append(boxes_list[i])
            res_img_list.append(img_list[i])
    print("Arr size:", len(res_boxes_list))
    write_pickle_local(arrays_filename, res_boxes_list)
    write_pickle_local(img_list_filename, res_img_list)


# TODO if needed complete this function. Predict boxes and clusterise results
# TODO Save cluster centers and load from this file and predict cluster inside
def save_box_cluster_members_label(df_name: str, images_dir: str,
                                   clusters_array_name: str, n_members: int,
                                   arrays_filename: str,
                                   img_list_filename: str,
                                   ignore_tag_ids: Optional[List[str]] = None,
                                   verbose: bool = True):
    df = read_df(df_name)
    images_path = DATASETS_DIR / images_dir

    img_list = defaultdict(list)
    boxes_list = defaultdict(list)
    for tag_id, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
        boxes = annotation_list_to_array(row.boxes)
        img_path = images_path / (PNG_FORMAT % tag_id)
        img = cv2.imread(str(img_path))
        num_boxes = len(boxes)
        img_list[num_boxes].append((tag_id, img))
        boxes_list[num_boxes].append(boxes)
    res_img_list = defaultdict(list)
    res_boxes_dict_of_lists = defaultdict(list)
    clusters_dict = read_pickle_local(clusters_array_name)
    for num_boxes, clusters in \
            clusters_dict.items():
        print("Num boxes:", num_boxes)
        n_clusters = len(np.unique(clusters))
        for cluster in range(n_clusters):

            cluster_members = np.where(clusters == cluster)[0]
            img_indices = np.random.choice(cluster_members, n_members + 1)
            cur_img_list = img_list[num_boxes]
            c = 0
            for i, (tag_id, _) in zip(img_indices, cur_img_list):
                if c == n_members:
                    break
                if ignore_tag_ids is not None and tag_id in ignore_tag_ids:
                    continue
                res_boxes_dict_of_lists[num_boxes].append(boxes_list[num_boxes][i])
                res_img_list[num_boxes].append(cur_img_list[i])
                c += 1
    print("Arr size:", len(res_boxes_dict_of_lists))
    write_pickle_local(arrays_filename, res_boxes_dict_of_lists)
    write_pickle_local(img_list_filename, res_img_list)


def save_box_cluster_members(df_name: str, images_dir: str,
                             clusters_array_name: str, n_members: int,
                             arrays_filename: str,
                             img_list_filename: str,
                             verbose: bool = True):
    df = read_df(df_name)
    images_path = DATASETS_DIR / images_dir

    img_list = defaultdict(list)
    boxes_list = defaultdict(list)
    for tag_id, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
        boxes = annotation_list_to_array(row.boxes)
        img_path = images_path / (PNG_FORMAT % tag_id)
        img = cv2.imread(str(img_path))
        num_boxes = len(boxes)
        img_list[num_boxes].append((tag_id, img))
        boxes_list[num_boxes].append(boxes)
    res_img_list = defaultdict(list)
    res_boxes_dict_of_lists = defaultdict(list)
    clusters_dict = read_pickle_local(clusters_array_name)
    for num_boxes, clusters in \
            clusters_dict.items():
        print("Num boxes:", num_boxes)
        n_clusters = len(np.unique(clusters))
        for cluster in range(n_clusters):
            cluster_members = np.where(clusters == cluster)[0]
            img_indices = np.random.choice(cluster_members, n_members)
            for i in img_indices:
                res_boxes_dict_of_lists[num_boxes].append(boxes_list[num_boxes][i])
                res_img_list[num_boxes].append(img_list[num_boxes][i])
    print("Arr size:", len(res_boxes_dict_of_lists))
    write_pickle_local(arrays_filename, res_boxes_dict_of_lists)
    write_pickle_local(img_list_filename, res_img_list)


def img_area(img):
    h, w = img.shape[:2]
    return h * w


def img_aspect_ratio(img):
    h, w = img.shape[:2]
    return w / h


def sample_from_histogram(img_dir_or_file: str,
                          id_list_filename: str, bins_area: List[float],
                          bins_ar: List[float], bin_tag_nums: List[int], base_dir: str,
                          n_clusters: int,
                          tag_nums_filename: str,
                          save_img_list_filename: Optional[str] = None):
    IMG_FORMAT = '.jpg'
    if save_img_list_filename is None:
        img_list = read_pickle_local(img_dir_or_file)
    else:
        img_dir_or_file = Path(img_dir_or_file)
        img_pathes = [img_path for img_path in img_dir_or_file.iterdir() if
                      img_path.is_file() and
                      img_path.suffix == IMG_FORMAT]
        num_imgs = len(img_pathes)
        img_ids = [img_path.name for img_path in img_pathes]
        img_list = [SkynetCV.load(str(img_path)) for img_path in tqdm(img_pathes,
                                                                     smoothing=.01,
                                                                     total=num_imgs)]
        write_pickle_local(save_img_list_filename, img_list)
        write_pickle_local(id_list_filename, img_ids)
    base_dir = Path(base_dir)
    id_files = read_pickle_local(id_list_filename)
    id_files = np.array(id_files)

    areas = np.array([img_area(img) for img in img_list])
    areas = np.log(areas)
    N = 111
    # res = sample_from_bins(areas, bins_area, id_files, N)
    # print("square", len(res))
    dir_square = Path(base_dir / 'square')
    dir_square.mkdir(exist_ok=True)
    # move_files(base_dir, dir_square, res)
    ignore_ids_mask = ignore_mask_from_dir(dir_square, id_files)

    img_ars = np.array([img_aspect_ratio(img) for img in img_list])
    N = 111
    # res = sample_from_bins(img_ars, bins_ar, id_files, N, ignore_ids_mask)
    # print("ar", len(res))
    dir_ar = Path(base_dir / 'ar')
    dir_ar.mkdir(exist_ok=True)
    # move_files(base_dir, dir_ar, res)
    ignore_ids_mask2 = ignore_mask_from_dir(dir_ar, id_files)
    print(np.sum(ignore_ids_mask & ignore_ids_mask2))
    ignore_ids_mask = ignore_ids_mask | ignore_ids_mask2

    # img_colors = np.array([np.mean(img, (0, 1)) for img in img_list])
    # clustering = cluster.KMeans(n_clusters, random_state=42).fit(img_colors)
    # # clustering = cluster.SpectralClustering(n_clusters, random_state=42).fit(img_colors)
    # print([(f"Cluster: {i}, num:{np.sum(clustering.labels_ == i)}") for i in range(
    #     n_clusters)])
    # N = 125
    # res = sample_from_bins(clustering.labels_, range(n_clusters), id_files, N,
    #                        ignore_ids_mask)
    # print("Clusters res len:", len(res))
    dir_color = Path(base_dir / 'color')
    dir_color.mkdir(exist_ok=True)
    # move_files(base_dir, dir_color, res)
    ignore_ids_mask2 = ignore_mask_from_dir(dir_color, id_files)
    print(np.sum(ignore_ids_mask & ignore_ids_mask2))
    ignore_ids_mask = ignore_ids_mask | ignore_ids_mask2

    try:
        tag_nums = read_pickle_local(tag_nums_filename)
    except FileNotFoundError:
        from ..recognizer import PriceRecognizer
        price_recognizer = PriceRecognizer()
        img_list = np.array(img_list)[~ignore_ids_mask]
        id_files = np.array(id_files)[~ignore_ids_mask]
        iter = zip(img_list, id_files)
        M = int(len(img_list) * 1.)
        # M = 5
        iter = random.sample(list(iter), M)
        iter = tqdm(iter, smoothing=.01)
        tag_nums = [(len(price_recognizer.detect(img)), filename) for img, filename in
                    iter]
        write_pickle_local(tag_nums_filename, tag_nums)

    N = 100
    tag_nums, id_files = list(zip(*tag_nums))
    # print(tag_nums)
    # print(id_files)
    res = sample_from_bins(np.array(tag_nums), bin_tag_nums, np.array(id_files), N)
    print("Digit count len:", len(res))

    dir_digit_count = Path(base_dir / 'digit_count')
    dir_digit_count.mkdir(exist_ok=True)
    print(dir_digit_count)
    move_files(base_dir, dir_digit_count, res)

    hist, bin_edges = np.histogram(areas, bins_area)
    hist2, bin_edges2 = np.histogram(img_ars, bins_ar)
    hist3, bin_edges3 = np.histogram(tag_nums, bin_tag_nums)
    return (hist, bin_edges), (hist2, bin_edges2), (hist3, bin_edges3)
    # return (hist, bin_edges), (hist2, bin_edges2), (hist3, bin_edges3), res
    # return (hist, bin_edges), (hist2, bin_edges2)


def ignore_mask_from_dir(dir_ar, id_files):
    ignore_ids_mask = np.zeros(len(id_files), dtype='bool')
    for p in dir_ar.iterdir():
        ignore_ids_mask[np.where(id_files == p.name)[0]] = True
    return ignore_ids_mask


def move_files(base_dir, dest_dir, filenames):
    for filename in filenames:
        shutil.copy(str(base_dir / 'raw' / filename), str(dest_dir / filename))


def sample_from_bins(x_list, bins, id_files, N, ignore_ids_mask=None):
    res = []
    ind = x_list <= bins[0]
    if ignore_ids_mask is not None:
        ind = ind & (~ignore_ids_mask)
    ind = np.nonzero(ind)[0]
    len_ind = len(ind)
    if len_ind:
        replace = len_ind < N
        ind = np.random.choice(ind, N, replace=replace)
        res.extend(list(id_files[ind]))
    for i in range(len(bins) - 1):
        ind = (bins[i] < x_list) & (x_list <= bins[i + 1])
        if ignore_ids_mask is not None:
            ind = ind & (~ ignore_ids_mask)
        ind = np.nonzero(ind)[0]
        len_ind = len(ind)
        if len_ind:
            replace = len_ind < N
            ind = np.random.choice(ind, N, replace=replace)
            res.extend(list(id_files[ind]))
    return res


def filter_dataset_by_blur(img_list_filename: str, boxes_arrays_filename: str,
                           blur_params: Optional[Dict[str, float]] = None,
                           ids: Optional[list] = None,
                           tqdm_enable: bool = True, show: bool = False):
    boxes_array_list = read_pickle_local(boxes_arrays_filename)
    img_list = read_pickle_local(img_list_filename)
    img_num = len(img_list)
    if ids is None:
        ids = range(img_num)
    res_boxes, res_imgs = [], []
    for i, (boxes_array, img) in tqdm(enumerate(zip(boxes_array_list, img_list)),
                                      total=img_num,
                                      disable=not tqdm_enable, smoothing=.01):
        if i not in ids:
            continue
        if blur_params is not None:
            blur_metric = calc_mean_blur_metrics(boxes_array, blur_params["dsize"], img)
            if blur_metric[3] < blur_params["threshold"]:
                continue
        if show: show_boxes_on_image(boxes_array, img)
        res_boxes.append(boxes_array)
        res_imgs.append(img)
    num = len(res_boxes)
    print("Not blur ratio", round(num / img_num * 100, 2), "Not blur num:", num)


def box_inside_predicate(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1[1:5]
    x1, y1 = box_center(xmax1, xmin1, ymax1, ymin1)
    xmin2, ymin2, xmax2, ymax2 = box2[1:5]
    return x1 > xmin2 and x1 < xmax2 and y1 > ymin2 and y1 < ymax2


def box_center(xmax1, xmin1, ymax1, ymin1):
    return (xmin1 + xmax1) / 2, (ymin1 + ymax1) / 2


def transform_annotations(anno_path: str, img_path, save_name: str, save_imgs: str):
    annotations = read_array(anno_path)
    imgs = read_array(img_path)
    new_imgs = []
    result = []
    count_zero_tags = 0
    for boxes, img in zip(annotations, imgs):
        digit_boxes, per_boxes, union_boxes, other_boxes = [], [], [], []
        for b in boxes:
            box_class = b[0]
            # print("box_class", box_class)
            if box_class <= 9:
                digit_boxes.append(b)
            elif box_class == 10:
                per_boxes.append(b)
            elif box_class == 13:
                union_boxes.append(b)
            else:
                other_boxes.append(b)
        tag_boxes = []
        if not other_boxes:
            continue
        for box in other_boxes:
            box_class = box[0]
            if box_class == 11:
                find_childs(box, digit_boxes, tag_boxes, 0)
            elif box_class == 12:
                find_childs(box, digit_boxes, tag_boxes, 1)
            elif box_class == 14:
                find_childs(box, per_boxes, tag_boxes, 2)
            elif box_class == 15:
                find_childs(box, per_boxes, tag_boxes, 3)
        # if per_boxes:
        #     print(f"digit_boxes: {digit_boxes},"
        #           f" per_boxes: {per_boxes}, tag_boxes: {tag_boxes}")
        #     show_boxes_on_image(digit_boxes, img)
        tag_id = 0
        result_union_boxes = []
        for box in union_boxes:
            tag_flag = False
            for i, child_box in enumerate(tag_boxes):
                if box_inside_predicate(child_box, box):
                    if len(child_box) == 7:
                        continue
                    tag_flag = True
                    tag_boxes[i] = np.append(child_box, tag_id)
            if tag_flag:
                result_union_boxes.append(np.array(np.append(box, tag_id)))
                tag_id += 1
        if tag_boxes:
            tag_boxes.extend(result_union_boxes)
            tag_boxes = np.array(tag_boxes)
            digit_boxes = np.array([b for b in tag_boxes if len(b) == 7 and b[0] < 10])
            if not len(digit_boxes):
                continue
            result.append(tag_boxes)
            new_imgs.append(img)
        else:
            count_zero_tags += 1
    print(f"count_zero_tags: {count_zero_tags}")
    write_array_local(save_name, np.array(result))
    write_array_local(save_imgs, np.array(new_imgs))


def find_childs(parent_box, boxes, result, child_class):
    childs = [(i, child_box) for i, child_box in \
              enumerate(boxes) if box_inside_predicate(child_box, parent_box)]
    for i, child in childs:
        child = np.append(child, child_class)
        result.append(np.array(child))


def create_box_features2(anno_name: str, img_path: str, save_name: str):
    annotations = read_array_local(anno_name)
    imgs = read_array(img_path)
    res = []
    for bboxes, img in tqdm(zip(annotations, imgs), total=len(annotations),
                            smoothing=.01):
        new_bboxes = np.array([x[1:5] for x in bboxes])
        img_h, img_w = img.shape[:2]
        features = boxes_features(new_bboxes, img_w, img_h)
        res.append(features)
    res = np.array(res)
    print(res.shape)
    write_array_local(save_name, res)


def generate_prices_array(labels_filename: str, img_filename: str, prices_filename: str):
    labels = read_array_local(labels_filename)
    imgs = read_array_local(img_filename)
    c = Counter()
    prices_list = []
    for img, boxes_list in zip(imgs, labels):
        digit_boxes = np.array([b for b in boxes_list if len(b) == 7 and b[0] < 10])
        try:
            max_tag_id = np.max(digit_boxes[:, 6]) + 1
        except IndexError as e:
            print(digit_boxes.shape)
            raise
        c[max_tag_id] += 1
        prices = []
        for tag_id in range(max_tag_id):
            digit_boxes_cur = digit_boxes[digit_boxes[:, 6] == tag_id]
            if not len(digit_boxes_cur):
                continue
            digit_boxes_cur = digit_boxes_cur[digit_boxes_cur[:, 1].argsort()]
            rub_digits = digit_boxes_cur[digit_boxes_cur[:, 5] == 0][:, 0]
            if not len(rub_digits):
                continue
            kop_digits = digit_boxes_cur[digit_boxes_cur[:, 5] == 1][:, 0]
            rub_float = digits_to_number(rub_digits)
            kop_float = digits_to_number(kop_digits) / 100
            price_float = round(rub_float + kop_float, 2)
            left_top_points = np.array(digit_boxes_cur[:, (1, 2)])
            x, y = left_top_points[:, 0].mean(), left_top_points[:, 1].mean()
            img_h, img_w = img.shape[:2]
            prices.append(((x / img_w, y / img_h), price_float))
        if len(prices) > 1:
            prices = sorted(prices, key=lambda x: (x[0][0], x[0][1]))
        prices_list.append(prices)
    prices_array = np.array(prices_list)
    print(prices_list[:10])
    print(len(prices_list))
    print("Num pricetags:", c.most_common())
    write_array_local(prices_filename, prices_array)


def min_max_boxes_to_center_boxes(boxes: np.ndarray):
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i]
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        boxes[i] = np.array([center_x, center_y, width, height])


def add_random_boxes(tag_box, rub_features, kop_features, res_labels, res, img_h,
                     img_w,
                     num_random_boxes: int, random_box_max_width_ratio: float,
                     random_box_max_height_ratio: float, img: np.ndarray, debug=False):
    random_box_min_width_ratio = .02
    random_box_min_width = int(random_box_min_width_ratio * img_w / 2)
    random_box_min_height_ratio = .05
    random_box_min_height = int(random_box_min_height_ratio * img_h / 2)
    random_box_min_height = max(random_box_min_height, 2)
    start_width_ratio = random_box_max_width_ratio
    start_width = int(start_width_ratio * img_w / 2)
    start_height_ratio = random_box_max_height_ratio
    start_height = max(int(start_height_ratio * img_h / 2), 4)
    xmin, ymin, xmax, ymax = tag_box[1:5]
    for _ in range(num_random_boxes):
        max_w = int(random_box_max_width_ratio * img_w)
        max_h = int(random_box_max_height_ratio * img_h)
        center_x = np.random.randint(start_width, img_w - start_width - 1)
        if center_x < xmin - start_width or xmax + start_width < center_x:
            if random_box_min_width < xmin:
                width = np.random.randint(random_box_min_width, min(xmin, max_w))
            else:
                continue
            if random_box_min_height < max_h:
                height = np.random.randint(random_box_min_height, max_h)
            else:
                continue
            center_y = np.random.randint(height + 1, img_h - height - 1)
        else:
            width = np.random.randint(random_box_min_width, max_w)
            high = ymin - start_height - 1
            if start_height < high:
                center_y = np.random.randint(start_height, high)
                height = np.random.randint(random_box_min_height, (ymin - center_y) // 2)
            else:
                continue
        new_box = np.array([center_x, center_y, width, height])
        samples = [np.concatenate((new_box, kop_features[:4])),
                   np.concatenate((new_box, kop_features[:8])),
                   np.concatenate((rub_features, new_box)),
                   np.concatenate((new_box, rub_features)),
                   np.concatenate((rub_features, kop_features[:4], new_box)),
                   np.concatenate((rub_features, kop_features[:8],
                                   new_box)),
                   np.concatenate((new_box, rub_features, kop_features[:4])),
                   np.concatenate((new_box, rub_features, kop_features[:8]))]
        rand_inds = np.random.randint(1, len(samples), 2, 'uint32')
        samples = [samples[i] for i in rand_inds]
        for sample_i in samples:
            sample_i = sample_i.astype('float32')
            normalize_features(img_h, img_w, sample_i)
            assert np.max(sample_i) <= 1., np.min(sample_i) >= 0.
            num_boxes = int(len(sample_i) / 4)
            res[num_boxes].append(sample_i)
            res_labels[num_boxes].append(num_boxes + 1)
        if debug:
            show_boxes_on_image(np.array([np.array(
                [center_x - width // 2, center_y - height // 2, center_x + width // 2,
                 center_y + height // 2])]), img)
            print("new_box", new_box)


def add_fp_boxes(fp_boxes, rub_boxes, kop_digits_features, res_labels, res,
                 img_h, img_w):
    len_fp_boxes = len(fp_boxes)
    inds = list(range(len_fp_boxes))
    random.shuffle(inds)
    shuffled_boxes = fp_boxes[inds][0, :, 1:5]
    min_max_boxes_to_center_boxes(shuffled_boxes)
    new_fp_boxes_list = []
    # print("shuffled_boxes:", len(shuffled_boxes))
    for new_box in shuffled_boxes:
        # print("new_box", new_box)
        new_fp_boxes_list.append(new_box)
        cur_fp_size = len(new_fp_boxes_list)
        if cur_fp_size > 1:
            sample_i = np.concatenate(tuple(new_fp_boxes_list))
            sample_i = sample_i.astype('float32')
            normalize_features(img_h, img_w, sample_i)
            # print("1", sample_i)
            assert np.max(sample_i) <= 1., np.min(sample_i) >= 0.
            assert cur_fp_size == len(sample_i) // 4
            res[cur_fp_size].append(sample_i)
            res_labels[cur_fp_size].append(cur_fp_size + 1)
        samples = [np.concatenate((rub_boxes, new_box)),
                   np.concatenate((new_box, rub_boxes))]
        if len(kop_digits_features):
            ext_list = [np.concatenate((new_box, kop_digits_features[:4])),
                        np.concatenate((new_box, kop_digits_features[:8])),
                        np.concatenate((rub_boxes, kop_digits_features[:4], new_box)),
                        np.concatenate((rub_boxes, kop_digits_features[:8], new_box)),
                        np.concatenate((new_box, rub_boxes, kop_digits_features[:4])),
                        np.concatenate((new_box, rub_boxes, kop_digits_features[:8]))]
            samples.extend(ext_list)
        num_random_boxes = 2
        rand_inds = np.random.randint(1, len(samples), num_random_boxes, 'uint32')
        samples = [samples[i] for i in rand_inds]
        for sample_i in samples:
            sample_i = sample_i.astype('float32')
            normalize_features(img_h, img_w, sample_i)
            # print("2", sample_i)
            assert np.max(sample_i) <= 1., np.min(sample_i) >= 0.
            num_boxes = int(len(sample_i) / 4)
            res[num_boxes].append(sample_i)
            res_labels[num_boxes].append(num_boxes + 1)


def match_iou(y_true: np.ndarray, y_pred: np.ndarray,
              gt_thresh: float):
    true_mask = np.ones((y_true.shape[0]), dtype=bool)
    pred_mask = np.ones((y_pred.shape[0]), dtype=bool)
    tp, fp = [], []
    tp_inds, fn_inds = set(), set()
    if len(y_true) and len(y_pred):
        similarities = iou(y_true[:, 1:5], y_pred[:, 1:], coords='corners')
        while True:
            max_iou_index = np.unravel_index(np.argmax(similarities), similarities.shape)
            max_iou = similarities[max_iou_index]
            if max_iou > gt_thresh:
                ind_x, ind_y = max_iou_index
                similarities[ind_x, :] = -1
                similarities[:, ind_y] = -1
                true_mask[ind_x] = False
                pred_mask[ind_y] = False
                if ind_y in tp_inds:
                    continue
                tp.append(np.append(y_pred[ind_y], y_true[ind_x][-2:]))
                tp_inds.add(ind_y)
            else:
                break
        fp.append(y_pred[pred_mask != 0])
    elif len(y_pred) and not len(y_true):
        fp = y_pred
    return np.array(tp), np.array(fp)


def create_price_features(anno_name: str, img_name: str,
                          save_features: str, save_labels: str,
                          pred_boxes_filename: str,
                          num_random_boxes: int = 20,
                          random_box_max_width_ratio: float = .3,
                          random_box_max_height_ratio: float = .3,
                          gt_thresh=.5,
                          debug=False):
    def true_boxes_add_samples(boxes_array, res,
                               res_labels, tag_boxes, img, tag_num, fp_boxes_array,
                               debug=False):
        for tag_id in range(tag_num):
            digit_boxes_cur = boxes_array[boxes_array[:, 6] == tag_id]
            tag_box = tag_boxes[tag_boxes[:, 5] == tag_id][0]
            if not len(digit_boxes_cur):
                continue
            digit_boxes_cur = np.unique(digit_boxes_cur, axis=0)
            digit_boxes_cur = digit_boxes_cur[digit_boxes_cur[:, 1].argsort()]
            rub_digits = digit_boxes_cur[digit_boxes_cur[:, 5] == 0]
            # if len(rub_digits):
            #     print(rub_digits.shape)
            #     show_boxes_on_image(rub_digits[:, :5], img)
            rub_num = len(rub_digits)
            if not rub_num:
                continue
            kop_digits = digit_boxes_cur[digit_boxes_cur[:, 5] == 1]
            kop_num = len(kop_digits)
            # debug = kop_num == 2 and rub_num == 2
            if debug:
                print("tag_id, rub_digits, kop_digits:", tag_id, rub_digits,
                      kop_digits, "rub num:", rub_num, "kop num:", kop_num)
            kop_digits_features = kop_digits[:, 1:5]
            min_max_boxes_to_center_boxes(kop_digits_features)
            rub_digits_features = rub_digits[:, 1:5]
            min_max_boxes_to_center_boxes(rub_digits_features)
            if debug:
                print("kop_digits_features width:", kop_digits_features,
                      "rub_digits_features:", rub_digits_features)
            kop_digits_features = kop_digits_features.flatten()

            rub_num = rub_num if kop_num else rub_num - 1
            for i in range(rub_num):
                if debug:
                    print("shapes", rub_digits_features.shape,
                          kop_digits_features.shape)
                rub_slice = rub_digits_features[i:].flatten()
                true_boxes_add_samples_helper(rub_slice.copy(),
                                              rub_digits_features.copy(),
                                              kop_digits_features.copy(), res_labels, res,
                                              fp_boxes_array, kop_num, i, img, debug)
                change_width_nums = 5
                for _ in range(change_width_nums):
                    max_width_change = .05
                    rub_slice = rub_slice.copy()
                    rub_digits_features = rub_digits_features.copy()
                    kop_digits_features = kop_digits_features.copy()
                    for j in range(2, len(rub_slice), 4):
                        max_width_change_pixel = int(max_width_change * rub_slice[j])
                        if max_width_change_pixel > 1:
                            if np.random.randint(0, 1):
                                translate = np.random.randint(1,
                                                              max_width_change_pixel)
                            else:
                                translate = np.random.randint(-max_width_change_pixel,
                                                              0)
                            rub_slice[j] += translate
                    # for j in range(3, len(rub_slice), 4):
                    #     max_width_change_pixel = int(max_width_change * rub_slice[j])
                    #     if max_width_change_pixel > 1:
                    #         if np.random.randint(0, 1):
                    #             translate = np.random.randint(1,
                    #                                           max_width_change_pixel)
                    #         else:
                    #             translate = np.random.randint(-max_width_change_pixel,
                    #                                           0)
                    #         rub_slice[j] += translate
                    true_boxes_add_samples_helper(rub_slice, rub_digits_features,
                                                  kop_digits_features, res_labels, res,
                                                  fp_boxes_array, kop_num, i, img, debug)

    def true_boxes_add_samples_helper(rub_slice, rub_digits_features, kop_digits_features,
                                      res_labels, res, fp_boxes_array, kop_num, i, img,
                                      debug):
        img_h, img_w = img.shape[:2]
        labels = [len(rub_digits_features) - i]
        if kop_num:
            samples = [np.concatenate((rub_slice, kop_digits_features))
                           .astype('float32')]
            # add_random_boxes(tag_box, rub_slice, kop_digits_features, res_labels,
            #                  res, img_h, img_w, num_random_boxes,
            #                  random_box_max_width_ratio,
            #                  random_box_max_height_ratio, img)
            if fp_boxes_array is not None and fp_boxes_array.shape[1]:
                add_fp_boxes(fp_boxes_array, rub_slice, [], res_labels, res,
                             img_h, img_w)
            if kop_num > 1:
                samples.append(np.concatenate(
                    (rub_slice, kop_digits_features[:4])).astype('float32'))
                labels.append(labels[-1])
        else:
            # add_random_boxes(tag_box, rub_slice, [], res_labels,
            #                  res, img_h, img_w, num_random_boxes,
            #                  random_box_max_width_ratio,
            #                  random_box_max_height_ratio, img)
            if fp_boxes_array is not None and fp_boxes_array.shape[1]:
                add_fp_boxes(fp_boxes_array, rub_slice, [], res_labels, res,
                             img_h, img_w)
            samples = [rub_slice]
        for sample_i in samples:
            normalize_features(img_h, img_w, sample_i)
        if debug:
            print("samples", samples, "label", labels)
        for sample_i, label in zip(samples, labels):
            assert np.max(sample_i) < 1., np.min(sample_i) > 0.
            num_boxes = int(len(sample_i) / 4)
            res[num_boxes].append(sample_i)
            res_labels[num_boxes].append(label)

    def pred_boxes_add_samples(tp_boxes_array, fp_boxes_array, res,
                               res_labels, img, tag_num, debug=False):
        for tag_id in range(tag_num):
            digit_boxes_cur = tp_boxes_array[tp_boxes_array[:, 6] == tag_id]
            if not len(digit_boxes_cur):
                continue
            digit_boxes_cur = np.unique(digit_boxes_cur, axis=0)
            digit_boxes_cur = digit_boxes_cur[digit_boxes_cur[:, 1].argsort()]
            rub_digits = digit_boxes_cur[digit_boxes_cur[:, 5] == 0]
            rub_num = len(rub_digits)
            if not rub_num:
                continue
            kop_digits = digit_boxes_cur[digit_boxes_cur[:, 5] == 1]
            if debug:
                show_boxes = np.concatenate((rub_digits, kop_digits), 0)[:, 1:5]
                print("concat", show_boxes.shape, fp_boxes_array)
                show_boxes_on_image(show_boxes, img)
            kop_num = len(kop_digits)
            kop_digits_features = kop_digits[:, 1:5]
            min_max_boxes_to_center_boxes(kop_digits_features)
            rub_digits_features = rub_digits[:, 1:5]
            min_max_boxes_to_center_boxes(rub_digits_features)
            kop_digits_features = kop_digits_features.flatten()

            rub_num = rub_num if kop_num else rub_num - 1
            for i in range(rub_num):
                rub_slice = rub_digits_features[i:].flatten()
                img_h, img_w = img.shape[:2]
                labels = [len(rub_digits_features) - i]
                if kop_num:
                    samples = [np.concatenate((rub_slice, kop_digits_features))
                                   .astype('float32')]
                    if fp_boxes_array.shape[1]:
                        add_fp_boxes(fp_boxes_array, rub_slice, kop_digits_features,
                                     res_labels, res, img_h, img_w)
                    if kop_num > 1:
                        samples.append(np.concatenate(
                            (rub_slice, kop_digits_features[:4])).astype('float32'))
                        labels.append(labels[-1])
                else:
                    # print("fp_boxes_array", fp_boxes_array, fp_boxes_array.shape,
                    #       len(fp_boxes_array))
                    if fp_boxes_array.shape[1]:
                        add_fp_boxes(fp_boxes_array, rub_slice, [], res_labels, res,
                                     img_h, img_w)
                    samples = [rub_slice]
                for sample_i in samples:
                    normalize_features(img_h, img_w, sample_i)
                for sample_i, label in zip(samples, labels):
                    assert np.max(sample_i) < 1., np.min(sample_i) > 0.
                    num_boxes = int(len(sample_i) / 4)
                    res[num_boxes].append(sample_i)
                    res_labels[num_boxes].append(label)

    annotations = read_array_local(anno_name)
    imgs = read_array_local(img_name)
    pred_boxes_array = read_pickle_local(pred_boxes_filename)
    max_side = 120
    res = defaultdict(list)
    res_labels = defaultdict(list)
    for bboxes, img, pred_boxes in tqdm(zip(annotations, imgs, pred_boxes_array),
                                        total=len(annotations), smoothing=.01):
        tag_boxes = np.array([b for b in bboxes if b[0] == 13])
        digit_boxes = [b for b in bboxes if len(b) == 7 and b[0] < 10]
        digit_boxes = np.array(digit_boxes)
        if len(pred_boxes):
            resize_to_detector_scale(pred_boxes, img, max_side)
        # show_boxes_on_image(pred_boxes, img)
        # show_boxes_on_image(digit_boxes[:, :5], img)
        if len(digit_boxes):
            max_tag_id = np.max(digit_boxes[:, -1]) + 1
        else:
            continue
        # tp_boxes_cur = tp_boxes[tp_boxes[:, 6] == tag_id]
        tp_boxes, fp_boxes = match_iou(digit_boxes, pred_boxes, gt_thresh)
        true_boxes_add_samples(digit_boxes, res, res_labels, tag_boxes, img,
                               max_tag_id, fp_boxes if len(fp_boxes) else None)
        # print(f"tp: {tp_boxes}\n fp: {fp_boxes}")
        if len(pred_boxes) and len(tp_boxes):
            pred_boxes_add_samples(tp_boxes, fp_boxes, res, res_labels, img,
                                   max_tag_id)

    for k, v in res.items():
        res[k] = np.array(v)
    write_pickle_local(save_features, res)
    write_pickle_local(save_labels, res_labels)


def create_price_features2(anno_name: str, img_name: str,
                           save_features: str, save_labels: str,
                           pred_boxes_filename: str,
                           gt_thresh=.5,
                           debug=False):
    annotations = read_array_local(anno_name)
    imgs = read_array_local(img_name)
    pred_boxes_array = read_pickle_local(pred_boxes_filename)
    max_side = 120
    res = defaultdict(list)
    res_labels = defaultdict(list)
    for bboxes, img, pred_boxes in tqdm(zip(annotations, imgs, pred_boxes_array),
                                        total=len(annotations), smoothing=.01):
        tag_boxes = np.array([b for b in bboxes if b[0] == 13])
        digit_boxes = [b for b in bboxes if len(b) == 7 and b[0] < 10]
        digit_boxes = np.array(digit_boxes)
        if len(pred_boxes):
            resize_to_detector_scale(pred_boxes, img, max_side)
        # show_boxes_on_image(pred_boxes, img)
        # show_boxes_on_image(digit_boxes[:, :5], img)
        if len(digit_boxes):
            max_tag_id = np.max(digit_boxes[:, -1]) + 1
        else:
            continue
        # tp_boxes_cur = tp_boxes[tp_boxes[:, 6] == tag_id]
        tp_boxes, fp_boxes = match_iou(digit_boxes, pred_boxes, gt_thresh)

    for k, v in res.items():
        res[k] = np.array(v)
    write_pickle_local(save_features, res)
    write_pickle_local(save_labels, res_labels)


def resize_to_detector_scale(boxes: np.ndarray, img: np.ndarray, max_side: int):
    img_h, img_w = img.shape[:2]
    if img_h > img_w:
        h = max_side
        w = int(h / img_h * img_w)
    else:
        w = max_side
        h = int(w / img_w * img_h)
    scale_y = img_h / h
    scale_x = img_w / w
    boxes[:, (1, 3)] = (boxes[:, (1, 3)] * scale_x).astype('uint16')
    boxes[:, (2, 4)] = (boxes[:, (2, 4)] * scale_y).astype('uint16')


def normalize_features(h, w, ar):
    for i in range(0, len(ar) - 1, 2):
        ar[i] /= w
    for j in range(1, len(ar), 2):
        ar[j] /= h
