import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sub = os.path.join(ROOT, "yolov5")
if str(sub) not in sys.path:
    sys.path.append(str(sub))  # add ROOT to PATH

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    strip_optimizer,
    xyxy2xywh,
)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.augmentations import letterbox
import numpy as np
from myutils import draw_rectange, put_text
from sort.sort import *


class ProductDetector:
    def __init__(
        self,
        weights,
        imgsz=(320, 320),
        conf_thresh=0.25,
        iou_thresh=0.45,
        device="",
        debug=True,
        datayaml="",
        max_det=1000,
        roi=None,
    ):
        self.__device = select_device(device)
        # (x,y,w,h)
        self.__model = DetectMultiBackend(
            weights, device=self.__device, dnn=False, data=datayaml, fp16=False
        )
        self.__stride, self.__names, self.__pt = (
            self.__model.stride,
            self.__model.names,
            self.__model.pt,
        )
        self.__imgsz = check_img_size(imgsz, s=self.__stride)  # check image size

        self.__roi = roi
        self.__max_det = max_det
        self.__conf_thresh = conf_thresh
        self.__iou_thresh = iou_thresh
        self.__debug = debug
        self.__mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.25)

    def reset_tracker(self):
        self.__mot_tracker.trackers = []
        self.__mot_tracker.frame_count = 0

    def set_roi(self, roi):
        self.__roi = roi

    @property
    def get_roi(self):
        return self.__roi

    @property
    def is_debug(self):
        return self.__debug

    def roi_img(self):
        roi = self.__roi
        return self.__frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]].copy()

    def __prepare_frame(self, frame):
        img = frame[
            self.__roi[1] : self.__roi[1] + self.__roi[3],
            self.__roi[0] : self.__roi[0] + self.__roi[2],
        ].copy()
        img = letterbox(img, self.__imgsz[0], self.__stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return img

    def process_frame(self, frame):
        im = self.__prepare_frame(frame)
        im = torch.from_numpy(im).to(self.__device)
        im = im.half() if self.__model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.__model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            self.__conf_thresh,
            self.__iou_thresh,
            None,
            False,
            max_det=self.__max_det,
        )

        det = pred[0]

        bboxes = []
        if len(det):
            det[:, :4] = scale_coords(
                im.shape[2:], det[:, :4], (self.__roi[3], self.__roi[2], 3)
            ).round()
            for *xyxy, conf, cls in reversed(det):
                xyxy[0] += self.__roi[0]
                xyxy[1] += self.__roi[1]
                xyxy[2] += self.__roi[0]
                xyxy[3] += self.__roi[1]
                bbox = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf]
                bbox = [x.cpu().detach().numpy().item(0) for x in bbox]
                bboxes.append(bbox)

        if not len(bboxes):
            track_bbs_ids = self.__mot_tracker.update()
        else:
            bboxes = np.array(bboxes)
            track_bbs_ids = self.__mot_tracker.update(bboxes)

        return track_bbs_ids

    def debug(self, frame, track_bbs_ids):
        for bbox in track_bbs_ids:
            bbox = list(map(int, bbox.tolist()))
            x1, y1, x2, y2, name_idx = bbox
            name = f"ID: {name_idx}"
            frame = draw_rectange(frame, (x1, y1), (x2, y2))
            frame = put_text(frame, name, (x1, y1 - 10))
        # for bbox in bboxes:
        #     conf = bbox[4]
        #     bbox = list(map(int, bbox[:4]))
        #     frame = draw_rectange(frame, tuple(bbox[:2]), tuple(bbox[2:]))
        #     frame = put_text(frame, "{:.2f}".format(conf), tuple(bbox[:2]))

        frame = draw_rectange(
            frame,
            (self.__roi[0], self.__roi[1]),
            (self.__roi[0] + self.__roi[2], self.__roi[1] + self.__roi[3]),
        )
        return frame
