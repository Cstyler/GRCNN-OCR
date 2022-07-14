import argparse
import json
import os
import sys
from pathlib import Path
import cv2
from time import time
import statistics

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sub = os.path.join(ROOT, "yolov5")
if str(sub) not in sys.path:
    sys.path.append(str(sub))  # add ROOT to PATH

from productdetector import ProductDetector
from myutils import read_config, put_text


class Detector:
    def __init__(self, pos=None, device="0"):
        detector_dir = "detector/"
        with open(f"{detector_dir}config.json") as json_file:
            self.config = json.load(json_file)
        self.detector = ProductDetector(
            device=device,
            roi=self.config[pos]["product roi"],
            datayaml=detector_dir + self.config["yaml_product"],
            weights=detector_dir + self.config["product_weights"],
        )

    def reset_tracker(self):
        self.detector.reset_tracker()

    @property
    def get_roi(self):
        return self.detector.get_roi

    def set_roi(self, pos):
        self.detector.set_roi(self.config[pos]["product roi"])

    def detect(self, frames):
        return [self.detector.process_frame(frame).astype(int) for frame in frames]


def run(source=None, save_path=ROOT / "runs/product", pos=None):
    config = read_config()
    detector = ProductDetector(
        device="cpu",
        roi=config[pos]["product roi"],
        datayaml=config["yaml_product"],
        weights=config["product_weights"],
    )

    _, filename = os.path.split(source)
    save_path = os.path.join(save_path, filename)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 480))
    cap = cv2.VideoCapture(source)

    fps = []
    i = 0
    os.makedirs("runs/detect/products/", exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        start = time()
        bboxes = detector.process_frame(frame)
        fps.append(1.0 / (time() - start))

        if detector.is_debug:
            frame = detector.debug(frame, bboxes)
            frame = put_text(frame, f"FPS: {statistics.mean(fps):.2f}", (0, 20))
            cv2.imwrite(f"runs/detect/products/{i}.jpg", frame)
            i += 1
            # cv2.imshow("detect", frame)
            # cv2.waitKey(1)

        out.write(frame)

    cap.release()
    out.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pos",
        type=str,
        default="POS60",
        help="camera position: POS71, POS65, POS60 and etc",
    )
    parser.add_argument("--source", type=str, help="file")
    opt = parser.parse_args()
    opt.source = "source/2022-02-10-160015-POS60-BO-Y242-(00-53-49_00-53-55).mp4"
    opt.pos = "POS60"
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
