from collections import Counter, defaultdict
from pathlib import Path
import pickle
from typing import Any, Callable, List, Optional, Tuple
import numpy as np

import pandas as pd

from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision import set_image_backend
import imagehash

set_image_backend("accimage")


class ProductsDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        csv_folder: str = "../csv/",
        loader: Callable[[str], Any] = default_loader,
        target_transform: Optional[Callable] = None,
        split_dataset=False,
        val=False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        df = pd.read_csv(csv_folder + "fruits_vegetables_filtered.csv")
        products_types_dict = dict(zip(df["Материал"], df["Вид"]))
        df = pd.read_csv(csv_folder + "main_products.csv")
        products_types_dict_main = dict(zip(df["Материал"], df["Classifier"]))
        products_types_dict.update(products_types_dict_main)
        if val:
            with open(Path(self.root).parent / "classes.pkl", "rb") as f:
                class_to_idx = pickle.load(f)
            samples = self.make_dataset_val(class_to_idx, products_types_dict)
            classes = list(class_to_idx.keys())
        else:
            samples, classes, class_to_idx = self.make_dataset(products_types_dict)
            with open(Path(self.root).parent / "classes.pkl", "wb") as f:
                pickle.dump(class_to_idx, f)
            if split_dataset:
                self.split_dataset(samples, classes)
        self.loader = loader
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

    def split_dataset(self, samples, classes):
        np.random.shuffle(samples)
        targets = np.array([y for _, y in samples])
        samples_per_class_counter = {
            i: (targets == i).sum() for i in range(len(classes))
        }
        classes_validation = Counter()
        MAX_SAMPLES_PER_CLASS = 60
        SAMPLES_PERCENT_PER_CLASS = 0.1
        MIN_SAMPLES_PER_CLASS = 10
        for path, y in samples:
            samples_num = samples_per_class_counter[y]
            if samples_num > MIN_SAMPLES_PER_CLASS and (
                classes_validation[y]
                < min(
                    MAX_SAMPLES_PER_CLASS, int(SAMPLES_PERCENT_PER_CLASS * samples_num)
                )
            ):
                classes_validation[y] += 1
                p = Path(path)
                new_p = Path(self.root).parent / "val"
                for part_p in p.parts[2:-1]:
                    new_p /= part_p
                new_p.mkdir(parents=True, exist_ok=True)
                new_p /= p.name
                p.rename(new_p)

    def make_dataset(
        self, products_types_dict, check_files=False
    ) -> List[Tuple[str, int]]:
        target_plus = sorted(products_types_dict.keys())
        root = Path(self.root)
        cls_idx = 0
        class_to_idx = {}
        instances = []
        if check_files:
            duplicates = Counter()
            wrong_files = Counter()
            remove_paths = []
        for plu in target_plus:
            plu_str = str(plu)
            plu_dir = root / plu_str[0] / plu_str[1] / plu_str[2] / plu_str
            if plu_dir.exists():
                if check_files:
                    hashes = set()
                for path_obj in plu_dir.iterdir():
                    path = str(path_obj)
                    if check_files:
                        try:
                            img = default_loader(path)
                            # low to high duplicate detection: dhash, dhash_vertical,
                            # whash, average_hash, phash_simple
                            hash_ = str(imagehash.dhash(img, hash_size=32))
                            if hash_ not in hashes:
                                hashes.add(hash_)
                            else:
                                remove_paths.append(path_obj)
                                duplicates[plu] += 1
                                continue
                        except:
                            remove_paths.append(path_obj)
                            wrong_files[plu] += 1
                            continue
                    class_name = products_types_dict[plu]
                    if class_name not in class_to_idx:
                        class_to_idx[class_name] = cls_idx
                        cls_idx += 1
                    instances.append((path, class_to_idx[class_name]))

        classes = sorted(class_to_idx.keys()) + ["Other"]
        count = 0
        class_to_idx["Other"] = cls_idx
        count_other = Counter()
        for d1 in root.iterdir():
            for d2 in d1.iterdir():
                for d3 in d2.iterdir():
                    for plu_dir in d3.iterdir():
                        if int(plu_dir.name) not in target_plus:
                            succeed = False
                            if check_files:
                                hashes = set()
                            for path_obj in plu_dir.iterdir():
                                path = str(path_obj)
                                if check_files:
                                    try:
                                        img = default_loader(path)
                                        hash_ = str(imagehash.dhash(img, hash_size=32))
                                        if hash_ not in hashes:
                                            hashes.add(hash_)
                                        else:
                                            remove_paths.append(path_obj)
                                            duplicates[plu] += 1
                                            continue
                                    except:
                                        remove_paths.append(path_obj)
                                        wrong_files[plu] += 1
                                        continue
                                succeed = True
                                instances.append((path, cls_idx))
                            if succeed:
                                count += 1
                                count_other[int(plu_dir.name)] += len(
                                    list(plu_dir.iterdir())
                                )
        if check_files:
            for p in remove_paths:
                p.unlink()
        idx_to_cls = {idx: cls for cls, idx in class_to_idx.items()}
        return instances, classes, class_to_idx

    def make_dataset_val(self, class_to_idx, products_types_dict):
        target_plus = sorted(products_types_dict.keys())
        root = Path(self.root)
        instances = []
        for plu in target_plus:
            plu_str = str(plu)
            plu_dir = root / plu_str[0] / plu_str[1] / plu_str[2] / plu_str
            if plu_dir.exists():
                for path_obj in plu_dir.iterdir():
                    path = str(path_obj)
                    class_name = products_types_dict[plu]
                    instances.append((path, class_to_idx[class_name]))
        count = 0
        count_other = Counter()
        other_cls_idx = len(class_to_idx) - 1
        for d1 in root.iterdir():
            for d2 in d1.iterdir():
                for d3 in d2.iterdir():
                    for plu_dir in d3.iterdir():
                        if int(plu_dir.name) not in target_plus:
                            succeed = False
                            for path_obj in plu_dir.iterdir():
                                path = str(path_obj)
                                succeed = True
                                instances.append((path, other_cls_idx))
                            if succeed:
                                count += 1
                                count_other[int(plu_dir.name)] += len(
                                    list(plu_dir.iterdir())
                                )
        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


if __name__ == "__main__":
    root = "dataset/train"
    val = False
    # root = "dataset/val"
    # val = True
    split_dataset = False
    d = ProductsDataset(root, csv_folder="csv/", split_dataset=split_dataset, val=val)
