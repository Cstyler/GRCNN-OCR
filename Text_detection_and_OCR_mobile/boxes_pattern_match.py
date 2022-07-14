import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from price_detector.data_processing import boxes_features
from price_detector.data_processing.utils import read_pickle_local


def calc_boxes_centers_x(box: np.ndarray) -> np.float:
    return (box[:, 1] + box[:, 3]) / 2


def calc_boxes_centers_y(box: np.ndarray) -> np.float:
    return (box[:, 2] + box[:, 4]) / 2


def calc_annotated_pattern(bboxes: np.ndarray):
    centers_x = calc_boxes_centers_x(bboxes)
    centers_y = calc_boxes_centers_y(bboxes)
    patterns = []
    for center_x, center_y, class_ in zip(centers_x, centers_y, bboxes[:, 0]):
        if class_:
            pattern = pattern_calc(center_x, center_y, centers_x, centers_y)
            patterns.append((pattern, class_))
    return patterns


def pattern_calc(center_x, center_y, centers_x, centers_y):
    return (np.count_nonzero(centers_x < center_x),
            np.count_nonzero(centers_x >= center_x),)
    # np.count_nonzero(centers_y < center_y),
    # np.count_nonzero(centers_y >= center_y))


def init_kmeans(cluster_centers_file):
    cluster_centers_dict = read_pickle_local(cluster_centers_file)
    res = {}
    for num_tags, cluster_centers in cluster_centers_dict.items():
        kmeans = KMeans()
        kmeans.cluster_centers_ = cluster_centers
        res[num_tags] = kmeans
    return res


def init_scaler(features_means_file, features_scales_file):
    features_means_dict = read_pickle_local(features_means_file)
    features_scales_dict = read_pickle_local(features_scales_file)
    res = {}
    for (num_tags, feature_means), feature_scales in zip(features_means_dict.items(),
                                                         features_scales_dict.values()):
        scaler = StandardScaler()
        scaler.mean_ = feature_means
        scaler.scale_ = feature_scales
        res[num_tags] = scaler
    return res


EMPTY_ARRAY = np.array([])


class PriceMatcher:
    def __init__(self, cluster_centers_file: str,
                 features_means_file: str,
                 features_scales_file: str,
                 annotated_cluster_members_file: str):
        self.kmeans = init_kmeans(cluster_centers_file)
        self.scaler = init_scaler(features_means_file, features_scales_file)
        self.annotated_cluster_members = read_pickle_local(annotated_cluster_members_file)

    def cluster_predict(self, boxes: np.ndarray, w: int, h: int):
        features = boxes_features(boxes[:, 1:], w, h)
        num_boxes = len(boxes)
        if num_boxes not in self.scaler:
            raise ValueError()
        features = self.scaler[num_boxes].transform([features])
        return self.kmeans[num_boxes].predict(features)[0]

    def match(self, boxes: np.ndarray, w: int, h: int, verbose=False):
        cluster = self.cluster_predict(boxes, w, h)
        # print("Cluster:", cluster, self.annotated_cluster_members[len(boxes)][cluster])
        _, _, annotated_boxes = self.annotated_cluster_members[len(boxes)][cluster]
        if len(annotated_boxes) != len(boxes):
            return
        annotated_pattern = calc_annotated_pattern(annotated_boxes)
        boxes_centers_x, boxes_centers_y = calc_boxes_centers_x(boxes), \
                                           calc_boxes_centers_y(boxes)
        rub_tuples, kop_tuples = [], []
        for box_center_x, box_center_y, digit in zip(boxes_centers_x, boxes_centers_y,
                                                     boxes[:, 0]):
            pattern = pattern_calc(box_center_x, box_center_y, boxes_centers_x,
                                   boxes_centers_y)
            for feature, class_ in annotated_pattern:
                if feature == pattern:
                    price_tuple = np.array((box_center_x, class_, digit))
                    if class_ == 1:
                        rub_tuples.append(price_tuple)
                    elif class_ == 2:
                        kop_tuples.append(price_tuple)
                    break
        if rub_tuples:
            rub_arr = np.array(rub_tuples)
            rub_arr = rub_arr[rub_arr[:, 0].argsort()]
            rub = rub_arr[:, 2]
        else:
            rub = EMPTY_ARRAY
        if kop_tuples:
            kop_arr = np.array(kop_tuples)
            kop_arr = kop_arr[kop_arr[:, 0].argsort()]
            kop = kop_arr[:, 2]
        else:
            kop = EMPTY_ARRAY
        return rub, kop
