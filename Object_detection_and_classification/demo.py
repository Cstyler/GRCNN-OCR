import os
from pathlib import Path
import pickle
import PIL
import imageio
import torchvision
import torch
import cv2
from torchvision.transforms import transforms
from detector.detect2 import Detector
from detector.productdetector import draw_rectange, put_text
from train.dataset import ProductsDataset


def main(webcam, model_path, gpu_flag):
    num_classes = 211  # TODO read from config
    model = "efficientnet_b3"
    model = torchvision.models.__dict__[model](num_classes=num_classes)
    device = torch.device("cuda" if gpu_flag else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.eval()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    trans = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    with open("dataset/classes.pkl", "rb") as f:
        cls_to_idx = pickle.load(f)
    idx_to_cls = {idx: cls for cls, idx in cls_to_idx.items()}
    detector = Detector(pos="POS60", device="0" if gpu_flag else "cpu")
    if webcam:
        detect_products(
            model,
            trans,
            idx_to_cls,
            detector,
            0,
            "result.mp4",
        )
    else:
        root = Path("test_dataset")
        save_dir = Path("test_dataset_results")
        save_dir.mkdir(exist_ok=True)
        for d1 in root.iterdir():
            for d2 in d1.iterdir():
                for d3 in d2.iterdir():
                    for mp4_dir in d3.iterdir():
                        for mp4_file in mp4_dir.iterdir():
                            save_file = save_dir / f"{mp4_file.stem}.mp4"
                            detect_products(
                                model,
                                trans,
                                idx_to_cls,
                                detector,
                                str(mp4_file),
                                str(save_file),
                            )


def detect_products(model, trans, idx_to_cls, detector, mp4_file, save_file):
    detector.reset_tracker()
    cap = cv2.VideoCapture(mp4_file)
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(frame)
    bboxes_list = detector.detect(images)
    writer = imageio.get_writer(save_file, fps=20)
    detected_ids = set()
    with torch.inference_mode():
        for frame, bboxes in zip(images, bboxes_list):
            for bbox in bboxes:
                bbox = list(map(int, bbox.tolist()))
                x1, y1, x2, y2, _ = bbox
                frame = draw_rectange(frame, (x1, y1), (x2, y2))
                crop = PIL.Image.fromarray(frame[y1:y2, x1:x2])
                res = model(torch.unsqueeze(trans(crop), 0))[0].argmax()
                res = int(res)
                detected_ids.add(res)
                frame = put_text(frame, str(res), (x1, y1 - 10))
            writer.append_data(frame)
    print("Classify", mp4_file)
    for id_ in detected_ids:
        print(id_, idx_to_cls[id_], flush=True)

if __name__ == "__main__":
    webcam = True
    # webcam = False
    # model_path = "train/checkpoints_old/model_75.pth"
    model_path = "train/model_30.pth"
    gpu_flag = False
    # gpu_flag = True
    main(webcam, model_path, gpu_flag)
