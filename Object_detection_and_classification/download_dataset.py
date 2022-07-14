from collections import Counter
import datetime
from functools import cached_property
import os
import time
from typing import Dict
from unicodedata import name
import uuid
import imageio
import numpy as np
import requests
from pathlib import Path
import re
import pandas as pd
from tempfile import NamedTemporaryFile

from detector.detect2 import Detector, ProductDetector

SECONDS_IN_HOUR = 3600
MAX_NUM_PER_CLASS = 50
MAX_NUM_OTHER_CLASSES = 6000


def main():
    df = pd.read_csv("csv/fruits_vegetables_filtered.csv")
    product_ids_fruits_vegs = set(df["Материал"].array)
    products_types_dict = dict(zip(df["Материал"], df["Вид"]))
    df = pd.read_csv("csv/main_products.csv")
    product_ids_main = set(df["Материал"].array)
    yandex_disk_url = "https://cloud-api.yandex.net/v1/disk/public/resources?public_key=https://disk.yandex.ru/d/hDNbj0U0sZen2w"
    yandex_disk_download = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/hDNbj0U0sZen2w"
    directories_names = get_directories_names(yandex_disk_url)
    res_fruits, res_main = (Counter(), Counter())
    count = 0
    video_flag = True
    # video_flag = False # TODO for counting
    device = "cpu"
    # device='0'
    if video_flag:
        detector = Detector(pos="POS60", device=device)
    other_plu_counter = Counter()
    other_classes_flag = True
    # save_dir = 'dataset/'
    save_dir = "test_dataset/"
    for dir_i, dir_name in enumerate(directories_names):
        if dir_i < 4:  # TODO
            continue
        dir_name_match = re.match(r"(POS\d+)", dir_name)
        if not dir_name_match:
            continue
        mp4_files, log_files = get_mp4_and_logs(yandex_disk_url, dir_name)
        if not log_files or not mp4_files:
            continue
        pos_type = dir_name_match.group(0)
        for log_j, (log_filename, log_timestamp) in enumerate(log_files):
            if not (log_timestamp.year == 2022 and log_timestamp.month == 6 and log_timestamp.day == 22):
                continue
            log_video_files = find_video_in_list(mp4_files, log_timestamp, False)
            if not log_video_files:
                continue
            log_lines = get_log_lines(yandex_disk_download, dir_name, log_filename)
            cur_video = None
            for plu_i, (crep_str, _, _, _, plu_str) in (
                (i, x) for i, x in enumerate(log_lines) if len(x[-1]) > 1
            ):
                time_stamp = extract_timestamp_from_crep_str(crep_str)
                plu = int(plu_str)
                if plu not in products_types_dict:
                    continue  # TODO
                if products_types_dict[plu] not in {
                    "томат свежий",
                    "'картофель свежий'",
                    "салат листовой",
                    "яблоко",
                    "огурцы свежие",
                    "банан",
                    "лук",
                    "морковь свежая",
                    "мандарин",
                    "лимон",
                    "салат листовой",
                    "грибы свежие",
                }:
                    continue  # TODO
                main_classes_flag = (
                    plu in product_ids_fruits_vegs or plu in product_ids_main
                )
                # main_classes_flag = False #TODO remove this is rerun on full classes
                if other_classes_flag and not main_classes_flag:
                    if sum(other_plu_counter.values()) > MAX_NUM_OTHER_CLASSES:
                        other_classes_flag = False
                        continue
                    if other_plu_counter[plu] > MAX_NUM_PER_CLASS:
                        continue
                    main_classes_flag = True
                if main_classes_flag:
                    if cur_video is not None:
                        time_delta = time_stamp - cur_video.start_time
                    if (
                        cur_video is None
                        or time_delta.days != 0
                        or SECONDS_IN_HOUR <= time_delta.seconds
                        or time_delta.seconds < 0
                    ):
                        video_info = find_video_in_list(log_video_files, time_stamp)
                        assert len(video_info) <= 5
                        if video_info:
                            video_file, video_datetime = video_info[0]
                            if video_flag:
                                if cur_video is not None:
                                    cur_video.close()
                                try:
                                    cur_video = Video(
                                        yandex_disk_download,
                                        dir_name,
                                        video_file,
                                        video_datetime,
                                        pos_type,
                                        detector,
                                    )
                                except:
                                    continue
                        else:
                            # did not found any mp4 related to this log
                            break
                    (
                        left_timestamp,
                        right_timestamp,
                        product_scanned,
                    ) = find_left_right_timestamps_of_video_crop(
                        plu_i, log_lines, time_stamp
                    )
                    if not product_scanned:
                        continue
                    saved = False
                    if video_flag:
                        try:
                            saved = cur_video.find_products_in_video(
                                left_timestamp,
                                right_timestamp,
                                time_stamp,
                                plu,
                                save_dir,
                            )
                        except:
                            continue
                    if saved:
                        count += 1
                    if plu in product_ids_fruits_vegs and saved:
                        res_fruits[products_types_dict[plu]] += 1
                    elif plu in product_ids_main and saved:
                        res_main[plu] += 1
                    elif saved:
                        other_plu_counter[plu] += 1
        print("Completed", dir_i, dir_name, count, flush=True)

    count_df = pd.DataFrame(list(res_fruits.items()), columns=("Вид", "Кол-во"))
    count_df.to_csv("csv/count_fruits_types.csv")
    count_df = pd.DataFrame(list(res_main.items()), columns=("Материал", "Кол-во"))
    count_df.to_csv("csv/count_main_types.csv")


def find_video_in_list(mp4_files, time_stamp, check_hour=True):
    if check_hour:
        timedelta_iter = (
            (name, video_datetime, time_stamp - video_datetime)
            for name, video_datetime in mp4_files
        )
        return [
            (name, video_datetime)
            for name, video_datetime, timedelta in timedelta_iter
            if 0 < timedelta.seconds <= SECONDS_IN_HOUR
        ]
    timedelta_iter = (
        (name, video_datetime, video_datetime - time_stamp)
        for name, video_datetime in mp4_files
    )
    return [
        (name, video_datetime)
        for name, video_datetime, timedelta in timedelta_iter
        if timedelta.days == 0
    ]


def find_left_right_timestamps_of_video_crop(
    log_j,
    log_lines,
    current_time: datetime.time,
    threshold_seconds=3,
    left_delta=datetime.timedelta(seconds=2, milliseconds=500),
    right_delta=datetime.timedelta(seconds=0, milliseconds=300),
):
    (left, product_scanned), right = find_left_right_indices_of_video_crop_helper(
        log_lines, current_time, threshold_seconds, range(log_j - 1, 0, -1), True
    ), find_left_right_indices_of_video_crop_helper(
        log_lines, current_time, threshold_seconds, range(log_j + 1, len(log_lines))
    )
    if product_scanned:
        timedelta = right - left
        if timedelta.seconds < threshold_seconds:
            right += right_delta
            left -= left_delta
    return left, right, product_scanned


NOT_SCANNED_ACTIONS_CODE = {"11901", "11902"}


def find_left_right_indices_of_video_crop_helper(
    log_lines, current_time, threshold_seconds, iter_range, check_product_scanned=False
):
    prev_timestamp = current_time
    product_scanned = True
    for j in iter_range:
        crep_str, _, _, _, plu_str = log_lines[j]
        action_code = crep_str.split(";")[2]
        if check_product_scanned and action_code in NOT_SCANNED_ACTIONS_CODE:
            product_scanned = False
        time_stamp = extract_timestamp_from_crep_str(crep_str)
        if plu_str or (current_time - time_stamp).seconds > threshold_seconds:
            if check_product_scanned:
                return prev_timestamp, product_scanned
            return prev_timestamp
        prev_timestamp = time_stamp
    if check_product_scanned:
        return prev_timestamp, product_scanned
    return prev_timestamp


def extract_timestamp_from_crep_str(crep_str):
    time_stamp = crep_str.split(";")[7]
    day = int(time_stamp[:2])
    month = int(time_stamp[2:4])
    year = int(time_stamp[4:8])
    hour = int(time_stamp[8:10])
    minute = int(time_stamp[10:12])
    sec = int(time_stamp[12:14])
    ms = int(time_stamp[14:17]) * 1000
    return datetime.datetime(year, month, day, hour, minute, sec, ms)


MIN_DIST_TO_ROI_BOTTOM = 0.1  # TODO find suitable values
ROI_LEFT_BORDER_RATIO = 0.55
ROI_RIGHT_BORDER_RATIO = 0.6


class BBox:
    def __init__(self, coords, roi):
        self.number_of_diffs = 0
        self.coords = coords
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi
        self.left_point, self.right_point = coords[:2], coords[2:4]
        self.left_bottom = self.left_point.copy()
        self.left_bottom[1] = self.right_point[1]
        self.right_bottom = self.right_point.copy()
        self.right_bottom[1] = self.left_point[1]

    @cached_property
    def left_border_roi(self):
        return int(self.roi_x + self.roi_w * ROI_LEFT_BORDER_RATIO)

    @cached_property
    def right_border_roi(self):
        return int(self.roi_x + self.roi_w * ROI_RIGHT_BORDER_RATIO)

    @cached_property
    def roi_bottom(self):
        return self.roi_y + self.roi_h

    @property
    def is_near_to_roi_bottom(self):
        # scanner is between 55% and 60% of roi width by x, product could be 3% up from roi by y
        lx, ly = self.left_bottom
        rx = self.right_bottom[0]
        return (abs(self.roi_bottom - ly) / self.roi_h < MIN_DIST_TO_ROI_BOTTOM) and (
            (
                lx < self.right_border_roi
                and self.left_border_roi < rx < self.right_border_roi
            )
            or (self.left_border_roi < lx < self.right_border_roi)
            or (lx < self.left_border_roi and rx > self.right_border_roi)
        )

    @property
    def left_right_points(self):
        return self.left_point, self.right_point, self.left_bottom, self.right_bottom

    def inc_diffs(self):
        self.number_of_diffs += 1

    def calc_diffs(self, next_bbox):
        (
            left_point2,
            right_point2,
            left_bottom2,
            right_bottom2,
        ) = next_bbox.left_right_points
        diffs = np.array(
            [
                self.calc_point_diff(self.left_point, left_point2),
                self.calc_point_diff(self.right_point, right_point2),
                self.calc_point_diff(self.left_bottom, left_bottom2),
                self.calc_point_diff(self.right_bottom, right_bottom2),
            ]
        )

        return diffs

    @cached_property
    def roi_diag(self):
        return np.linalg.norm([self.roi_w, self.roi_h])

    def calc_point_diff(self, point1, point2):
        return np.linalg.norm(point1 - point2) / self.roi_diag


SEARCH_SCAN_START_RATIO = 0.1  # TODO debug this (maybe find scan index using timestamp)


class Video:
    def __init__(
        self,
        yandex_disk_download,
        dir_name,
        mp4_file,
        video_datetime,
        pos_type,
        detector: ProductDetector,
    ):
        video_content = download_file_from_yandex_disk(
            yandex_disk_download, dir_name, mp4_file
        ).content
        self.tempfile = NamedTemporaryFile(suffix=".mp4", dir="/dev/shm")
        self.tempfile.write(video_content)
        self.vid = imageio.get_reader(self.tempfile.name, "ffmpeg")
        self.fps = self.vid.get_meta_data()["fps"]
        self.mp4_file = Path(mp4_file)
        self.start_time = video_datetime
        self.pos_type = pos_type
        self.video_counter = 0
        detector.set_roi(pos_type)
        self.detector = detector

    def find_products_in_video(
        self,
        left_timestamp: datetime.time,
        right_timestamp: datetime.time,
        scan_timestamp: datetime.time,
        plu: int,
        save_dir: str,
        debug=1,  # TODO
    ):
        left_ind = self.get_video_ind(left_timestamp)
        right_ind = self.get_video_ind(right_timestamp)
        # scan_ind = self.get_video_ind(scan_timestamp) - left_ind
        images = [self.vid.get_data(i) for i in range(left_ind, right_ind)]
        plu = str(plu)
        plu_dir = Path(f"{save_dir}{plu[0]}/{plu[1]}/{plu[2]}/{plu}")
        # plu_dir = Path(f"dataset/{plu}")  # TODO only for debug
        self.detector.reset_tracker()
        bboxes_list = self.detector.detect(images)
        if debug:
            plu_dir.mkdir(exist_ok=True, parents=True)
            writer = imageio.get_writer(
                plu_dir / f"{self.mp4_file.stem}_{self.video_counter}.mp4", fps=self.fps
            )
            self.video_counter += 1
            for img, bboxes in zip(images, bboxes_list):
                # writer.append_data(self.detector.detector.debug(img, bboxes))
                writer.append_data(img)
        # return self.crop_products_from_video_fragment(bboxes_list, images, plu_dir) #TODO

    def get_video_ind(self, timestamp: datetime.time):
        if timestamp < self.start_time:
            raise ValueError(
                f"Timestamp of log is lower then timestamp of video:"
                f" {timestamp}, {self.start_time}"
            )
        time_delta = timestamp - self.start_time
        return int((time_delta.seconds + time_delta.microseconds / 1000000) * self.fps)

    def close(self):
        self.tempfile.close()
        assert not os.path.exists(self.tempfile.name)

    def get_scanned_products(self, bboxes_list):
        roi = self.detector.get_roi
        start_ind = int(SEARCH_SCAN_START_RATIO * len(bboxes_list))
        for bboxes in bboxes_list[start_ind:]:
            for bbox in bboxes:
                coords, name_idx = bbox[:-1], bbox[-1]
                if BBox(coords, roi).is_near_to_roi_bottom:
                    return [
                        [bbox for bbox in bboxes if bbox[-1] == name_idx]
                        for bboxes in bboxes_list
                    ]

    def crop_products_from_video_fragment(self, bboxes_list, images, save_dir):
        scanned_products = self.get_scanned_products(bboxes_list)
        if scanned_products is None:
            return
        saved = False
        for bboxes, img in zip(scanned_products, images):
            if not saved:
                save_dir.mkdir(exist_ok=True, parents=True)
            # det_id = str(uuid.uuid4())[:5]  # TODO for debug
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2, _ = bbox
                crop = img[y1:y2, x1:x2]
                # save_path = (
                #     save_dir
                #     / f"{self.mp4_file.stem}_{self.video_counter}_{det_id}_{i}.png"
                #     # TODO for debug
                # )
                det_id = str(uuid.uuid4())
                save_path = save_dir / f"{det_id}.png"
                imageio.imwrite(save_path, crop)
                saved = True
        return saved

    def filter_motionless_products(self, bboxes_list, threshold_diff=0.01):
        motion_less_dict: Dict[int, BBox] = {}
        skip_idx_set = set()
        roi = self.detector.get_roi
        for i, bboxes in enumerate(bboxes_list):
            for bbox in bboxes:
                name_idx = bbox[-1]
                if name_idx in skip_idx_set:
                    continue
                if name_idx not in motion_less_dict:
                    motion_less_dict[name_idx] = BBox(bbox, roi)
                else:
                    prev_bbox = motion_less_dict[name_idx]
                    if prev_bbox.number_of_diffs > 1:
                        del motion_less_dict[name_idx]
                        skip_idx_set.add(name_idx)
                        continue
                    diffs = prev_bbox.calc_diffs(BBox(bbox, roi))
                    # doesnt work if bbox randomly flucatuates around stable product
                    if all(diffs > threshold_diff):
                        prev_bbox.inc_diffs()
        motion_less_dict = {
            k: box for k, box in motion_less_dict.items() if box.is_near_to_roi_bottom
        }
        motionless_keys = motion_less_dict.keys()
        # motion_objects = [
        #     [bbox for bbox in bboxes if bbox[-1] not in motionless_keys]
        #     for bboxes in bboxes_list
        # ]
        for bboxes in bboxes_list:
            for bbox in bboxes:
                name_idx = bbox[-1]
                bbox = BBox(bbox, roi)
                if name_idx not in motionless_keys and bbox.is_near_to_roi_bottom:
                    pass


def get_log_lines(yandex_disk_download, dir_name, log_filename):
    resp_download = download_file_from_yandex_disk(
        yandex_disk_download, dir_name, log_filename
    )
    resp_download.encoding = resp_download.apparent_encoding
    log_file = resp_download.text
    plu_list = re.findall(
        r"(строка CREP.*)(\n  (PRODUCT_NAME.*\n)?(  BARCODE.*\n)?  PLU (\d+))?",
        log_file,
    )
    return plu_list


def download_file_from_yandex_disk(yandex_disk_download, dir_name, filename):
    resp = requests.get(f"{yandex_disk_download}&path=/{dir_name}/{filename}")
    log_file_url = resp.json()["href"]
    try:
        resp_download = requests.get(log_file_url)
    except:
        time.sleep(0.02)
        try:
            resp_download = requests.get(log_file_url)
        except:
            time.sleep(0.02)
            resp_download = requests.get(log_file_url)
    return resp_download


def get_directories_names(yandex_disk_url):
    resp = requests.get(yandex_disk_url)
    directories_items = get_items(resp)
    directories_names = [x["name"] for x in directories_items]
    return directories_names


def get_mp4_and_logs(yandex_disk_url, dir_name):
    get_url = f"{yandex_disk_url}&path=/{dir_name}&limit=9999"
    resp = requests.get(get_url)
    if not resp.ok:
        raise Exception(f"Request excepption: {resp.content}")
    dir_items = get_items(resp)
    file_names = [Path(x["name"]) for x in dir_items]
    mp4_files = []
    for mp4_file in (x.name for x in file_names if x.suffix == ".mp4"):
        # Example: 1_2022-03-16-145305-POS65-BO-Y242.mp4
        m = re.match(r".*(20\d{2})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})", mp4_file)
        try:
            mp4_files.append(
                (mp4_file, datetime.datetime(*(int(m.group(i)) for i in range(1, 7))))
            )
        except AttributeError:
            continue
    log_files = []
    for log_file in (x.name for x in file_names if x.suffix == ".log"):
        # Example: POS602-BO-B327_202206080000.log
        m = re.match(r".+_(20\d{2})(\d{2})(\d{2})\d+", log_file)
        try:
            log_files.append(
                (log_file, datetime.datetime(*(int(m.group(i)) for i in range(1, 4))))
            )
        except AttributeError:
            continue
    return mp4_files, log_files


def get_items(resp):
    return resp.json()["_embedded"]["items"]


if __name__ == "__main__":
    main()
