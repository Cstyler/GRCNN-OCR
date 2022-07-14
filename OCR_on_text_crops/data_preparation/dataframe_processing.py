from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Callable

import cytoolz
import pandas as pd
import tqdm
import ujson as json

from pylibs import cutter, img_utils, jpeg_utils, pandas_utils, rect_utils
from pylibs.google_recognizer import Recognizer
from pylibs.pandas_utils import DF_FILE_FORMAT
from pylibs.storage_utils import get_file_sharding, get_tag_from_share, save_file_sharding

PRODUCT_CODE_FIELD = 'ProductCode'
# ALPHABET = "".join(map(str, range(10))) + '-'
ALPHABET = "".join(map(str, range(10)))
INDEX_NAME = 'tag_id'


def process(dataset_dir_path: str, df_name: str, processed_df_name: str, img_dir: str,
            w_padding: float, h_padding: float,
            debug: bool = False):
    dataset_dir_path = Path(dataset_dir_path)
    processed_images_dir_path = dataset_dir_path / img_dir

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    processed_df = process_df(df_path, processed_images_dir_path, w_padding, h_padding, debug)
    df_save_path = dataset_dir_path / (DF_FILE_FORMAT % processed_df_name)
    if not debug:
        pandas_utils.write_dataframe(processed_df, df_save_path)


def process_df(df_path: Path,
               processed_img_dir_path: Path, w_padding: float, h_padding: float,
               debug: bool):
    rows = []
    df = pandas_utils.read_dataframe(df_path)
    for tag_id, row in tqdm.tqdm_notebook(df.iterrows(), total=len(df.index)):
        values = json.loads(row['values'])
        text = values[PRODUCT_CODE_FIELD]
        rows.append((tag_id, text))
        photo_id = row['photo.id']
        img_path = get_tag_from_share(tag_id, photo_id)
        save_img_path = save_file_sharding(processed_img_dir_path, img_path)
        if not save_img_path.exists():
            img = jpeg_utils.read_jpeg(img_path)
            segments = json.loads(row['segments'])
            product_code_segment = segments[PRODUCT_CODE_FIELD]
            unirect = rect_utils.UniversalRect.from_coords_dict(product_code_segment)
            unirect = unirect.add_padding(w_padding, h_padding)
            segment_img = cutter.cut_segment_from_img(img, unirect)
            jpeg_utils.write_jpeg(save_img_path, segment_img)
            if debug:
                img_utils.show_img(img)
                img_utils.show_img(segment_img)
                print(unirect)
                print(text)

    df = pd.DataFrame(rows, columns=(INDEX_NAME, 'text'))
    df.set_index(INDEX_NAME, inplace=True)
    return df


def post_process_labels_model_grcnn(dataset_dir_path: str, df_names: Iterable[Tuple[str, str]], max_len: int,
                                    outlier_len: int):
    chars = ALPHABET
    char_set = set(chars)
    n_classes = len(chars)
    chars2num = {x: i for i, x in enumerate(chars)}

    def extend_blank(iterable: Iterable[int], len_: int) -> List[int]:
        extend_size = max_len - len_
        return list(iterable) + [n_classes] * extend_size

    def process_df(df_path: Path, df_name: Optional[str] = None):
        rows = []
        df = pandas_utils.read_dataframe(df_path)
        for tag_id, row in df.iterrows():
            text = row['text']
            text_list = [x for x in text if x in char_set]
            len_vals = len(text_list)
            if not len_vals or len_vals > outlier_len:
                continue
            feature = extend_blank(map(chars2num.__getitem__, text_list), len_vals)
            new_row = tag_id, json.dumps(feature), len_vals, "".join(text)
            rows.append(new_row)
        new_df = pd.DataFrame(rows, columns=(INDEX_NAME, 'label', 'label_len', 'text'))
        if df_name:
            df_path = df_path.with_name(df_name)
        new_df.set_index(INDEX_NAME, inplace=True)
        pandas_utils.write_dataframe(new_df, df_path)

    dataset_dir_path = Path(dataset_dir_path)
    for df_name, save_df_name in df_names:
        df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
        process_df(df_path, DF_FILE_FORMAT % save_df_name)


def post_process_labels_model_freq(dataset_dir_path: str, df_name: str, new_df_name: str):
    chars = tuple(map(str, range(10)))

    def calc_char_freq(s: str) -> List[int]:
        char_count = Counter(s)
        return [char_count[c] for c in chars]

    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    rows = [(seg_id, json.dumps(calc_char_freq(row['text']))) for seg_id, row in df.iterrows()]
    new_df = pd.DataFrame(rows, columns=(INDEX_NAME, 'char_freq'))
    new_df.set_index(INDEX_NAME, inplace=True)
    pandas_utils.write_dataframe(new_df, df_path.with_name(DF_FILE_FORMAT % new_df_name))


def update_dataset_with_google_annotations(dataset_dir_path: str, df_name: str, save_df_name: str,
                                           divide_size: int = 14, img_dir="processed_images"):
    dataset_dir_path = Path(dataset_dir_path)
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    images_dir_path = dataset_dir_path / img_dir
    df = pandas_utils.read_dataframe(df_path)
    recognizer = Recognizer()

    src_type = 'file'
    new_rows = []

    google_text_col = 'google_text'
    if google_text_col in df:
        df.drop(google_text_col, axis=1, inplace=True)
    for i, batch in enumerate(cytoolz.partition_all(divide_size, df.iterrows())):
        sources = (get_file_sharding(images_dir_path, tag_id) for tag_id, _ in batch)
        results = recognizer.recognize_concrete_data(sources, src_type, 'code')
        for (tag_id, row), (google_text, _) in zip(batch, results):
            new_row = (tag_id,) + tuple(row) + (google_text,)
            new_rows.append(new_row)
    new_df = pd.DataFrame(new_rows, columns=(INDEX_NAME,) + tuple(df.columns) + (google_text_col,))
    new_df.set_index(INDEX_NAME, inplace=True)

    df_path = df_path.with_name(DF_FILE_FORMAT % save_df_name)
    pandas_utils.write_dataframe(new_df, df_path)


def add_column_to_df(dataset_dir_path: str, df_name: str, df_name2: str, key: str):
    dataset_dir_path = Path(dataset_dir_path)

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df1 = pandas_utils.read_dataframe(df_path)
    df1 = pd.DataFrame(df1.loc[:, key])

    df_path2 = dataset_dir_path / (DF_FILE_FORMAT % df_name2)
    df2 = pandas_utils.read_dataframe(df_path2)

    new_df = df2.merge(df1, on=INDEX_NAME)
    pandas_utils.write_dataframe(new_df, df_path2)

def rename_column(df: pd.DataFrame, *,
                  col_name: str, new_col_name: str) -> pd.DataFrame:
    return df.rename(columns={col_name: new_col_name})


def modify_df(dataset_dir_path: str, df_name: str, modify_fun: Callable[[pd.DataFrame], pd.DataFrame],
              save_df_name: Optional[str] = None):
    dataset_dir_path = Path(dataset_dir_path)
    new_df_name = DF_FILE_FORMAT % (save_df_name if save_df_name else df_name)
    df_name = DF_FILE_FORMAT % df_name
    df_path = dataset_dir_path / df_name
    df = pandas_utils.read_dataframe(df_path)
    df = modify_fun(df)
    pandas_utils.write_dataframe(df, dataset_dir_path / new_df_name)


def filter_df_by_img_existence(dataset_dir_path: str, df_name: str, img_dir: str, save_df_name: str):
    dataset_dir_path = Path(dataset_dir_path)
    img_dir = dataset_dir_path / img_dir
    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)

    indices = []
    for tag_id in df.index:
        img_path = get_file_sharding(img_dir, tag_id)
        if img_path.exists():
            indices.append(tag_id)

    new_df = df.loc[indices]
    new_df_path = df_path.with_name(DF_FILE_FORMAT % save_df_name)
    pandas_utils.write_dataframe(new_df, new_df_path)
