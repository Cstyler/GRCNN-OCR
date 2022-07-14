---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import sampling
import image_processing
import dataframe_processing
from pylibs import via_utils
from pathlib import Path
import functools

dataset_dir_path = Path('/srv/data_science/storage/product_code_ocr')
```

```python
df_name = 'noise'
processed_df_name = 'noise_processed'
img_dir = "processed_images"
w_padding, h_padding = 1.15, 1.00
debug = False
```

```python
dataframe_processing.process(dataset_dir_path, df_name, processed_df_name, w_padding, h_padding, debug=debug)
```

```python
df_name = 'fine'
processed_df_name = 'fine_processed'
dataframe_processing.process(dataset_dir_path, df_name, processed_df_name, w_padding, h_padding, debug=debug)
```

```python
df_name = 'oct_2019_fine'
processed_df_name = 'oct_2019_fine_processed'
img_dir = "oct_2019_processed_images"
w_padding, h_padding = 1.15, 1.00
debug = False
```

```python
dataframe_processing.process(dataset_dir_path, df_name, processed_df_name, img_dir, w_padding, h_padding, debug=debug)
```

```python
df_name = 'noise_processed'
num = 5200
save_df_name = 'noise_relabel'
```

```python
sampling.subsample_dataset(dataset_dir_path, df_name, num, save_df_name)
```

```python
df_name = 'noise_relabel'
image_dir_path = '/srv/data_science/storage/product_code_ocr/processed_images'
task_name = 'task2'
save_dir = f'/srv/data_science/storage/product_code_ocr/labeling_tasks/{task_name}'
file_attributes_key = ()
df_column_names = ()
divide_size = 300
values_from_json = False
image_save_dir = f'/opt/share-http/product_code_ocr/{task_name}'
photo_id_sharding = False
img_url_format = f'https://app.stg.metacommerce.ru/data-science/product_code_ocr/{task_name}/' + '{tag_id}.jpg'
```

```python
# df_name = 'test_relabel_11_20_len26'
df_name = 'val_relabel_11_20_len26'
image_dir_path = '/srv/data_science/storage/product_code_ocr/processed_images'
task_name = 'task7'
save_dir = f'/srv/data_science/storage/product_code_ocr/labeling_tasks/{task_name}'
file_attributes_key = ('Код продукта', )
df_column_names = ('text', )
divide_size = 300
values_from_json = False
image_save_dir = f'/opt/share-http/product_code_ocr/{task_name}'
photo_id_sharding = False
img_url_format = f'https://app.stg.metacommerce.ru/data-science/product_code_ocr/{task_name}/' + '{tag_id}.jpg'
```

```python
via_utils.export_labels_to_via(
    dataset_dir_path,
    df_name,
    image_dir_path,
    save_dir,
    file_attributes_key,
    df_column_names,
    divide_size,
    values_from_json=values_from_json,
    img_url_format=img_url_format,
    image_save_dir=image_save_dir,
    photo_id_sharding=photo_id_sharding,
)
```

```python
df_name = 'fine_processed'
image_dir_path = '/srv/data_science/storage/product_code_ocr/processed_images'
save_dir = '/srv/data_science/storage/product_code_ocr/labeling_tasks/task1'
file_attributes_key = ('Код продукта', )
df_column_names = ('text', )
divide_size = 300
values_from_json = False
image_save_dir = '/opt/share-http/product_code_ocr/task1'
photo_id_sharding = False
img_url_format = 'https://app.stg.metacommerce.ru/data-science/product_code_ocr/task1/{tag_id}.jpg'
```

```python
# relabel val
df_name = 'train_10k_relabel'
image_dir_path = '/srv/data_science/storage/product_code_ocr/cropped_images'
save_dir = '/srv/data_science/storage/product_code_ocr/labeling_tasks/task5'
file_attributes_key = ('Код продукта', )
df_column_names = ('text', )
divide_size = 300
values_from_json = False
image_save_dir = '/opt/share-http/product_code_ocr/task5'
photo_id_sharding = False
img_url_format = 'https://app.stg.metacommerce.ru/data-science/product_code_ocr/task5/{tag_id}.jpg'
```

```python
via_utils.export_labels_to_via(
    dataset_dir_path,
    df_name,
    image_dir_path,
    save_dir,
    file_attributes_key,
    df_column_names,
    divide_size,
    values_from_json=values_from_json,
    img_url_format=img_url_format,
    image_save_dir=image_save_dir,
    photo_id_sharding=photo_id_sharding,
)
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr/labeling_results/task2'
# file_attributes_key = ('Код продукта', )
file_attributes_key = ()
segment_key = 'rect'
# segment_key = None
# df_column_names = ('text', )
df_column_names = ()
# save_df_name = 'fine_processed_via'
save_df_name = 'train_rectified'
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr/labeling_results/task5'
file_attributes_key = ('Код продукта', )
segment_key = None
df_column_names = ('text', )
save_df_name = 'train_relabel_11_20'
```

```python
via_utils.import_labels_from_via(dataset_dir_path, file_attributes_key, df_column_names, save_df_name, segment_key)
```

```python
df_name = 'train_rectified'
img_dir = "images_rectified"
max_len = 26
train_df_name = f'train_set_len{max_len}'
save_df_name = f'rectified_set_len{max_len}'
debug = False
image_processing.crop_images(dataset_dir_path, df_name, iсеmg_dir, train_df_name, save_df_name, debug=debug)
```

```python
df_name = "noise_processed"
# w, h = 200, 32
# w, h = 128, 32
w, h = 100, 32
# img_dir = 'processed_images'
img_dir = 'cropped_images'
# save_img_dir = "images_train3"
save_img_dir = f'cropped_images_train_size{w}x{h}'
debug = False
image_processing.resize_images(dataset_dir_path, df_name, img_dir, save_img_dir, w, h, debug=debug)
df_name = "fine_processed"
image_processing.resize_images(dataset_dir_path, df_name, img_dir, save_img_dir, w, h)
```

```python
max_len = 26
df_name = f"rectified_set_len{max_len}"
# w, h = 200, 32
# w, h = 128, 32
w, h = 100, 32
save_img_dir = f'images_rectified_size{w}x{h}'
img_dir = 'images_rectified'
debug = False
image_processing.resize_images(dataset_dir_path, df_name, img_dir, save_img_dir, w, h, debug=debug)
```

```python
df_name = 'test_set_len26'
width, height = 100, 32

img_dir = 'cropped_images'
save_img_dir = f'images_train_padding_size{width}x{height}'
debug = False
image_processing.resize_images_padding(dataset_dir_path, df_name, img_dir, save_img_dir, width, height, debug)
```

```python
test_size = .04
max_len = 26
df_name = f"rectified_set_len{max_len}"
train_name = f"rectified_train_set_len{max_len}"
val_name = f"rectified_val_set_len{max_len}"
sampling.split_dataset(dataset_dir_path, df_name, test_size, train_name, val_name)
```

```python
max_len = 26
fine_df_name = f'fine_val_len{max_len}'
val_set_name = f'val_set_len{max_len}'
test_set_name = f'test_set_len{max_len}'
test_size = .5
sampling.split_dataset(dataset_dir_path, fine_df_name, test_size, val_set_name, test_set_name)
```

```python
# df_name = 'noise'
# old_df_name = 'noise3'
# df_name = 'fine'
# old_df_name = 'fine3'
df_name = 'fine'
old_df_name = 'oct_2019_fine'
image_dir_path = '/srv/data_science/storage/product_code_ocr/processed_images'
debug = False
image_processing.remove_old_imgs(dataset_dir_path, df_name, old_df_name, image_dir_path, debug)
```

```python
df_name = "oct_2019_fine"
df_name2 = "oct_2019_fine_processed2"
key = "photo.metaInfo.marketId"
dataframe_processing.add_column_to_df(dataset_dir_path, df_name, df_name2, key)
```

```python
max_text_len = 26
df_name = "fine"
df_name2 = f'test_set_len{max_text_len}'
key = "photo.metaInfo.marketId"
dataframe_processing.add_column_to_df(dataset_dir_path, df_name, df_name2, key)
```

```python
max_text_len = 26
df_name = f'test_set_len{max_text_len}'
modify_fun = functools.partial(dataframe_processing.rename_column, col_name='photo.metaInfo.marketId', new_col_name='market_id')
dataframe_processing.modify_df(dataset_dir_path, df_name, modify_fun)
```

```python
# df_name = f'train_set_len{max_text_len}_filtered_2'
# df_name2 = f'train_set_len{max_text_len}_filtered'
df_name = f'val_set_len{max_text_len}_filtered_2'
df_name2 = f'val_set_len{max_text_len}_filtered'
key = 'char_freq'
dataframe_processing.add_column_to_df(dataset_dir_path, df_name, df_name2, key)
```

```python
df_name = 'test_set'
divide_size = 14
save_df_name = 'test_set_google_bs_%s' % divide_size
dataframe_processing.update_dataset_with_google_annotations(dataset_dir_path, df_name, save_df_name, divide_size)
```

```python
import dataframe_processing, image_processing
from pathlib import Path
dataset_dir_path = Path('/srv/data_science/storage/product_code_ocr')
```

```python
max_len = 26
w, h = 100, 32
# df_name =  f'train_set_len{max_len}'
# save_df_name =  f'train_set_len{max_len}_filtered'
# df_name =  f'val_set_len{max_len}'
# save_df_name =  f'val_set_len{max_len}_filtered'
df_name =  f'test_set_len{max_len}'
save_df_name =  f'test_set_len{max_len}_filtered'
img_dir = f'images_train_padding_size{w}x{h}'
dataframe_processing.filter_df_by_img_existence(dataset_dir_path, df_name, img_dir, save_df_name)
```

```python
max_len = 26
df_names = (('noise_processed', f'train_set_len{max_len}'), (f'test_set_len{max_len}', f'test_set_len{max_len}'), (f'val_set_len{max_len}', f'val_set_len{max_len}'))
outlier_len = 18
dataframe_processing.post_process_labels_model_grcnn(dataset_dir_path, df_names, max_len, outlier_len)
```

```python
import dataframe_processing, image_processing
from pathlib import Path
dataset_dir_path = Path('/srv/data_science/storage/product_code_ocr')
```

```python
max_len = 26
df_names = [(f'train_relabel_11_20_len{max_len}', f'train_relabel_11_20_len{max_len}_2'),
           (f'val_relabel_11_20_len{max_len}', f'val_relabel_11_20_len{max_len}_2'),
           (f'test_relabel_11_20_len{max_len}', f'test_relabel_11_20_len{max_len}_2')]
```

```python
for df_name, new_df_name in df_names:
    dataframe_processing.post_process_labels_model_freq(dataset_dir_path, df_name, new_df_name)
```

```python
max_text_len = 26
# new_df_name = f'train_set_len{max_text_len}_filtered_2'
# df_name = f'train_set_len{max_text_len}_filtered'
new_df_name = f'val_set_len{max_text_len}_filtered_2'
df_name = f'val_set_len{max_text_len}_filtered'
```

```python
dataframe_processing.post_process_labels_model_freq(dataset_dir_path, df_name, new_df_name)
```

```python
key = "char_freq"
for df_name2, df_name in df_names:
    dataframe_processing.add_column_to_df(dataset_dir_path, df_name, df_name2, key)
```

```python
# max_len = 51
max_len = 26
# max_len = 50
# max_len = 33
# df_names = (('noise_processed', f'train_set_len{max_len}'), )
df_names = [('fine_processed_via', f'fine_val_len{max_len}')]
# df_names = (('oct_2019_fine_processed', f'oct_2019_fine_processed2'), )
outlier_len = 18
dataframe_processing.post_process_labels_model_grcnn(dataset_dir_path, df_names, max_len, outlier_len)
```

```python
max_len = 26
df_names = [
    ("train_relabel_11_20", f"train_relabel_11_20_len26"),
    ("val_relabel_11_20", f"val_relabel_11_20_len26"),
    ("test_relabel_11_20", f"test_relabel_11_20_len26"),
]
outlier_len = 18
dataframe_processing.post_process_labels_model_grcnn(
    dataset_dir_path, df_names, max_len, outlier_len
)
```

```python

```
