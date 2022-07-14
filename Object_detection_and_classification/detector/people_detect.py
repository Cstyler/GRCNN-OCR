import argparse
import os
from myutils import read_config
from pathlib import Path
import cv2
from peopledetector import PeopleDetector

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def run(source=None,
        save_path=ROOT / "runs/people",
        pos=None):
    config = read_config()

    detector = PeopleDetector(backgroundimage=cv2.imread(config[pos]['backgroundimage']),
                              roi=config[pos]['people roi'],
                              weights=config['empty_weights'],
                              occupancy_thresh=float(config[pos]['threshold']),
                              blur=True)

    _, filename = os.path.split(source)
    save_path = os.path.join(save_path, filename)

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        detector.process_frame(frame)
        if detector.is_debug:
            frame = detector.debug(frame)
            fgmask = detector.mask
            img = detector.roi_img
            if fgmask is not None:
                cv2.imshow("fgmask", fgmask)
            cv2.imshow("MOG2", frame)
            cv2.imshow("part", img)
            cv2.waitKey(30)


        out.write(frame)

    cap.release()
    out.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', type=str, default='POS60', help='camera position: POS71, POS65, POS60 and etc')
    parser.add_argument('--source', type=str, help='file')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
