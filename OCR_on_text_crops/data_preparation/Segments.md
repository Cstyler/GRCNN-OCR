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
from segments import process, rm_non_exist_rows
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr'
cache_base_dir = '/srv/data_science/cached_data/product_code_ocr'
max_side = 224
df_name = 'train_rectified'
source_img_dir = 'processed_images'
ds_dir_name = 'segmentation'
test_size = .04
```

```python
process(dataset_dir_path, cache_base_dir, max_side, df_name, source_img_dir, ds_dir_name, test_size)
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr'
w, h = 100, 32
img_dir = f'cropped_images_train_size{w}x{h}'
max_len = 26
```

```python
df_name = f'train_set_len{max_len}'
save_df_name = f'train_set_len{max_len}_filtered'
rm_non_exist_rows(dataset_dir_path, df_name, img_dir, save_df_name)
```

```python
df_name = f'val_set_len{max_len}'
save_df_name = f'val_set_len{max_len}_filtered'
rm_non_exist_rows(dataset_dir_path, df_name, img_dir, save_df_name)
```

```python
from pylibs.jpeg_utils import read_jpeg
from pylibs.img_utils import show_img
from pylibs.numpy_utils import read_array, print_stats
from pathlib import Path
```

```python
base_dir = Path(cache_base_dir) / 'segmentation' / 'test'
img_dir = base_dir / 'images'
lbl_dir = base_dir / 'labels'
lbls = sorted(lbl_dir.iterdir())
imgs = sorted(img_dir.iterdir())
```

```python
index = 10
img_path = imgs[index]
img = read_jpeg(img_path)
lbl_path = lbls[index]
lbl = read_array(lbl_path)
```

```python
show_img(img)
```

```python
show_img(lbl)
```

```python

```
