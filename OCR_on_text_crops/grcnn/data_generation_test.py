from itertools import islice
from pathlib import Path

from pylibs import img_utils
from pylibs.pandas_utils import DF_FILE_FORMAT, read_dataframe
from .data_generation import BatchGenerator
from .training import get_transforms

WHITE_PIXEL = 255


def main(df_name, img_dir, filter_fun):
    transform_list = get_transforms()
    dataset_dir_path = Path('/srv/data_science/storage/product_code_ocr')
    images_dir_path = dataset_dir_path / img_dir
    batch_size = 10
    max_text_len = 26

    val_set_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    val_df = read_dataframe(val_set_path)
    val_df = val_df.iloc[[i for i, x in enumerate(val_df.text) if filter_fun(x)]]
    val_gen = BatchGenerator(val_df, batch_size, images_dir_path, max_text_len, transform_list, seed=42,
                             augment_prob=1)
    for batch_x, _ in islice(val_gen, 10):
        for img, label in zip(batch_x['input'], batch_x['texts']):
            if filter_fun(label):
                img_utils.show_img((img + 1) / 2, convert2rgb=False)
                print(label)
