from pathlib import Path

from pylibs import pandas_utils
from pylibs.pandas_utils import DF_FILE_FORMAT
from .predictor import ALPHABET
from .training import BatchGenerator
from .utils import calc_metric, load_model_grcnn, set_gpu


def filter_bad_samples(dataset_dir_path: str, df_name: str, save_df_name: str,
                       model_num: int, epoch: int, max_text_len: int,
                       batch_size: int, img_dir: str,
                       height: int, width: int, n_classes: int,
                       grcl_niter: int, grcl_fsize: int, lstm_units: int):
    gpu_id = 1080
    dataset_dir_path = Path(dataset_dir_path)
    images_dir_path = dataset_dir_path / img_dir

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    test_gen = BatchGenerator(df, batch_size, images_dir_path, max_text_len, shuffle=False)

    set_gpu(gpu_id)

    model = load_model_grcnn(model_num, epoch, height, width, n_classes, max_text_len, grcl_fsize, grcl_niter,
                             lstm_units)

    _, predictions = model.predict_generator(test_gen, verbose=1)
    test_gen = BatchGenerator(df, 1, images_dir_path, max_text_len, shuffle=False)
    indices = []
    for (batch_x, _), prediction, (tag_id, row) in zip(test_gen, predictions, df.iterrows()):
        true_text = row['text']
        if not true_text:
            continue
        true_text = "".join(x for x in true_text if x in ALPHABET)
        dist = calc_metric(prediction, true_text, ALPHABET, False)
        if not dist:
            indices.append(tag_id)

    new_df = df.loc[indices]
    new_df_path = df_path.with_name(DF_FILE_FORMAT % save_df_name)
    pandas_utils.write_dataframe(new_df, new_df_path)