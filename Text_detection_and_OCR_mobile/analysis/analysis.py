from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from price_detector.data_processing import BBOX_KEYS, Dict, defaultdict
from price_detector.data_processing.numpy_utils import print_stats, read_array
from price_detector.data_processing.utils import annotation_list_to_array, \
    read_array_local, read_df, read_pickle_local, write_pickle_local
from price_detector.detector.box_utils import box_areas, box_aspect_ratios
from price_detector.detector.utils import show_boxes_on_image

MAIN_DIR = Path('..')
DATASETS_DIR = MAIN_DIR / 'datasets'
PNG_FORMAT = "%s.png"
np.random.seed(42)


def analyse_boxes(df_name: str, verbose: bool = False):
    df = read_df(df_name)
    bbox_keys = BBOX_KEYS[1:]

    area_ratios_total, aspect_ratios_total = [], []
    for tag_id, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
        area = row.img_w * row.img_h
        bboxes = annotation_list_to_array(row.boxes, bbox_keys)
        area_ratios = box_areas(bboxes) / area
        area_ratios_total.append(area_ratios)
        aspect_ratios = box_aspect_ratios(bboxes)
        aspect_ratios_total.append(aspect_ratios)

    area_ratios_total = np.concatenate(area_ratios_total)
    aspect_ratios_total = np.concatenate(aspect_ratios_total)

    print_stats(area_ratios_total * 100)
    print_stats(aspect_ratios_total)
    print(np.corrcoef(area_ratios_total, aspect_ratios_total))


def clusterisation(box_features_filename: str,
                   save_cluster_centers: str,
                   save_means: str,
                   save_scales: str, save_labels: str,
                   n_clusters: int) -> Dict[int, np.ndarray]:
    random_state = 42
    box_features = read_array_local(box_features_filename)
    box_features = np.array(box_features)
    standard_scaler = StandardScaler().fit(box_features)
    x_transformed = standard_scaler.transform(box_features)
    kmeans = KMeans(n_clusters, random_state=random_state).fit(x_transformed)
    write_pickle_local(save_means, standard_scaler.mean_)
    write_pickle_local(save_scales, standard_scaler.scale_)
    write_pickle_local(save_cluster_centers, kmeans.cluster_centers_)
    write_pickle_local(save_labels, kmeans.labels_)


def clusterisation_by_num(df_name: str,
                          save_cluster_centers: str,
                          save_means: str,
                          save_scales: str) -> Dict[int, np.ndarray]:
    feature_by_num_boxes = read_pickle_local(df_name)
    random_state = 42
    means, stds, centers = {}, {}, {}
    labels = {}
    for num_boxes, box_features in feature_by_num_boxes.items():
        box_features = np.array(box_features)
        standard_scaler = StandardScaler().fit(box_features)
        means[num_boxes] = standard_scaler.mean_
        stds[num_boxes] = standard_scaler.scale_
        x_transformed = standard_scaler.transform(box_features)
        n_clusters = num_tag_to_num_templates(num_boxes)
        kmeans = KMeans(n_clusters, random_state=random_state).fit(x_transformed)
        centers[num_boxes] = kmeans.cluster_centers_
        labels[num_boxes] = kmeans.labels_
    write_pickle_local(save_means, means)
    write_pickle_local(save_scales, stds)
    write_pickle_local(save_cluster_centers, centers)
    return labels


def show_box_clusters(df_name: str, images_dir: str,
                      clusters_array_name: str, n_members: int,
                      verbose: bool = True):
    df = read_df(df_name)
    images_path = DATASETS_DIR / images_dir
    bbox_keys = BBOX_KEYS[1:]

    imgs = defaultdict(list)
    boxes_list = defaultdict(list)
    for tag_id, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
        boxes = annotation_list_to_array(row.boxes, bbox_keys)
        img_path = images_path / (PNG_FORMAT % tag_id)
        img = cv2.imread(str(img_path))
        num_boxes = len(boxes)
        imgs[num_boxes].append(img)
        boxes_list[num_boxes].append(boxes)
    clusters_dict = read_pickle_local(clusters_array_name)
    for num_boxes, clusters in \
            clusters_dict.items():
        print("Num boxes:", num_boxes)
        n_clusters = len(np.unique(clusters))
        for cluster in range(n_clusters):
            cluster_members = np.where(clusters == cluster)[0]
            clust_member_num = len(cluster_members)
            img_indices = np.random.choice(cluster_members, n_members)
            print("Cluster:", cluster, "Num members:", clust_member_num)
            for i in img_indices:
                print(i)
                show_boxes_on_image(boxes_list[num_boxes][i], imgs[num_boxes][i])


def show_box_clusters2(anno_name: str, img_path: str,
                       clusters_array_name: str, n_members: int):
    boxes_list = read_array_local(anno_name)
    img_list = read_array(img_path)
    clusters = read_pickle_local(clusters_array_name)
    n_clusters = len(np.unique(clusters))
    print(len(img_list), len(boxes_list))
    print(clusters)
    for cluster in range(n_clusters):
        cluster_members = np.where(clusters == cluster)[0]
        clust_member_num = len(cluster_members)
        img_indices = np.random.choice(cluster_members, n_members)
        print("Cluster:", cluster, "Num members:", clust_member_num)
        for i in img_indices:
            print(i)
            if len(boxes_list[i].shape) > 1:
                show_boxes_on_image(boxes_list[i][:, :5], img_list[i])


MAX_NUM_BOXES = 15


def num_tag_to_num_templates(num_boxes):
    # Based on datasets statistics: (num_boxes, num_tags)
    # [(3, 3194), (5, 2180), (2, 634), (4, 396), (6, 240), (8, 227), (7, 93),
    # (10, 90), (11, 39), (1, 34), (9, 17), (13, 12), (15, 7), (14, 4)]
    # First if covers 89% of dataset.
    if num_boxes == 2:
        return 12
    elif num_boxes == 3:
        return 10
    elif num_boxes == 4:
        return 20
    elif num_boxes == 5:
        return 30
    elif num_boxes in {6, 7, 8, 10}:
        return 15
    elif num_boxes in {11, 1, 9, 13}:
        return 7
    elif 14 <= num_boxes <= 15:
        return 2
    raise ValueError(f"Max {MAX_NUM_BOXES} tags")
