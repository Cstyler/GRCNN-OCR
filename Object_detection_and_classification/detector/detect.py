# python detect.py --source 'source/2022-02-08-080000-POS71-BO-5194-(00-12-28_00-13-31).mp4' --weights models/weights/best.pt --pos POS71

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sub = os.path.join(ROOT, "yolov5")
if str(sub) not in sys.path:
    sys.path.append(str(sub))  # add ROOT to PATH

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (
    LOGGER,
    check_img_size,
    check_imshow,
    check_requirements,
    cv2,
    increment_path,
    non_max_suppression,
    scale_coords,
)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.augmentations import letterbox
import numpy as np
from myutils import read_config, draw_rectange
from sort.sort import *


def processing(img0, img_size=320, stride=None, roi=(109, 0, 338, 323)):
    img = img0[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]].copy()
    img = letterbox(img, img_size, stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img


@torch.no_grad()
def run(
    weights=ROOT / "yolov5s.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob, 0 for webcam
    data=ROOT / "product.yaml",  # dataset.yaml path
    imgsz=(320, 320),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=True,  # show results
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    visualize=True,  # visualize features
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    pos="",
):
    source = str(source)
    save_img = True and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    # Directories
    save_dir = increment_path(
        Path(project) / name, exist_ok=exist_ok, mkdir=True
    )  # increment run

    mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

    roi = read_config()[pos]["product roi"]
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    ## return path, img, img0, self.cap, s
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = processing(im0s, img_size=imgsz, stride=stride, roi=roi)
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = (
            increment_path(save_dir / Path(path).stem, mkdir=True)
            if visualize
            else False
        )
        pred = model(im, augment=False, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += "%gx%g " % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            bboxes = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                #                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], (roi[3], roi[2], 3)
                ).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy[0] += roi[0]
                    xyxy[1] += roi[1]
                    xyxy[2] += roi[0]
                    xyxy[3] += roi[1]
                    bbox = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf]
                    bbox = [x.cpu().detach().numpy().item(0) for x in bbox]
                    bboxes.append(bbox)
                    # if save_img or view_img:  # Add bbox to image
                    #     c = int(cls)  # integer class
                    #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #     annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            if not len(bboxes):
                track_bbs_ids = mot_tracker.update()
            else:
                bboxes = np.array(bboxes)
                track_bbs_ids = mot_tracker.update(bboxes)
            # im0 = annotator.result()
            im0 = draw_rectange(
                im0, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3])
            )
            for j in range(len(track_bbs_ids)):
                coords = track_bbs_ids.tolist()[j]
                x1, y1, x2, y2, name_idx = list(map(int, coords))
                name = f"ID: {name_idx}"
                color = (0, 0, 255)
                im0 = draw_rectange(im0, (x1, y1), (x2, y2))
                cv2.putText(
                    im0, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )

            # true_img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = im0
            # im0 = true_img
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(
                            Path(save_path).with_suffix(".mp4")
                        )  # force *.mp4 suffix on results source
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}Done. ({t3 - t2:.3f}s)")

    # Print results
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "product.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[320],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(
        "--pos",
        type=str,
        default="default",
        help="camera position: POS71, POS65, POS60 and etc",
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    opt.source = "source/2022-02-10-160015-POS60-BO-Y242-(00-53-49_00-53-55).mp4"
    opt.pos = "POS60"
    opt.visualize = False
    opt.view_img = False
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
