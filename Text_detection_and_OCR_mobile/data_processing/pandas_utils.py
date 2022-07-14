from pathlib import Path
from typing import Union

import pandas as pd


def read_dataframe(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    engine = kwargs.setdefault('engine', 'auto')
    return pd.read_parquet(path, engine=engine)


def write_dataframe(df: pd.DataFrame, path: Union[str, Path], **kwargs):
    engine = kwargs.setdefault('engine', 'auto')
    compression = kwargs.setdefault('compression', 'gzip')
    index = kwargs.setdefault('index', True)
    df.to_parquet(path, engine=engine, compression=compression, index=index)


def add_column_to_df(dataset_dir_path: str, df_name: str, df_name2: str, key: str):
    dataset_dir_path = Path(dataset_dir_path)

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df1 = read_dataframe(df_path)
    df1 = pd.DataFrame(df1.loc[:, key])

    df_path2 = dataset_dir_path / (DF_FILE_FORMAT % df_name2)
    df2 = read_dataframe(df_path2)

    new_df = df2.merge(df1, on=df2.index.name)
    write_dataframe(new_df, df_path2)


DF_FILE_FORMAT = "%s.parquet.gzip"
