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
from grcnn import metrics
from grcnn.utils import find_best_model_epoch
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr'
max_text_len = 26
w, h = 100, 32
n_classes = 10

grcl_fsize = 3
# grcl_niter = 3
grcl_niter = 3
lstm_units = 512

# df_name = f'test_set_len{max_text_len}'
df_name = f'test_relabel_11_20_len{max_text_len}'
# df_name = f'val_set_len{max_text_len}'
# df_name = f'train_set_len{max_text_len}'
model_num = 51
epoch = find_best_model_epoch(model_num)
# epoch = 99 # m46
# epoch = 16 # m37
# epoch = 29 # m20
# epoch = 58 # m21
# epoch = 38 # m18

batch_size = 80
# image_dir = "images_train"
# image_dir = "images_train2"
# image_dir = "images_train3"
# image_dir = "images_train3"
# image_dir = f'cropped_images_train_size{w}x{h}' 
image_dir = f'images_train_padding_size{w}x{h}'
full_img_dir = 'processed_images'
dist_filter = lambda x: x > 0
debug = 1
```

```python
sign(x-y)
```

```python
metrics.test_nn(dataset_dir_path, df_name, model_num, epoch, batch_size, w, h, max_text_len,
                n_classes, grcl_niter, grcl_fsize, lstm_units,
                image_dir, full_img_dir, debug, dist_filter)
```

```python
# d = {14: (3, 79), 13: (1267, 1457), 7: (411, 432), 6: (561, 606), 9: (67, 67), 10: (32, 33), 15: (0, 5), 16: (0, 2), 12: (3, 16), 8: (20, 22)} # m7e80

# d = {14: (3, 79), 13: (1287, 1457), 7: (414, 432), 6: (559, 606), 9: (67, 67), 10: (33, 33), 15: (0, 5), 16: (0, 2), 12: (3, 16), 8: (20, 22)} # m8e103 test
# d = {14: (3, 79), 13: (1286, 1457), 7: (417, 432), 6: (560, 606), 9: (66, 67), 10: (33, 33), 15: (0, 5), 16: (0, 2), 12: (4, 16), 8: (20, 22)} # m8e165 test

# d = {7: (18452, 20092), 9: (9377, 9510), 6: (39518, 42663), 8: (1606, 1836), 10: (1543, 1775), 11: (58, 200), 13: (72288, 77363), 12: (859, 3415), 5: (382, 445), 14: (224, 923), 15: (338, 562), 16: (65, 126), 17: (2, 6)} # m8e103 train
# d = {14: (1, 79), 13: (1270, 1457), 7: (426, 432), 6: (594, 606), 9: (65, 67), 10: (33, 33), 15: (0, 5), 16: (0, 2), 12: (3, 16), 8: (19, 22)} # m16e32
# d = {14: (3, 79), 13: (1261, 1457), 7: (430, 432), 6: (593, 606), 9: (67, 67), 10: (33, 33), 15: (0, 5), 16: (0, 2), 12: (3, 16), 8: (19, 22)} # m22e54
# d = {14: (3, 79), 13: (1270, 1457), 7: (429, 432), 6: (599, 606), 9: (67, 67), 10: (33, 33), 15: (0, 5), 16: (0, 2), 12: (2, 16), 8: (19, 22)} # m37e16
# d = {13: (1359, 1394), 7: (425, 428), 6: (591, 599), 14: (70, 97), 9: (66, 66), 10: (33, 33), 8: (18, 20), 15: (4, 6), 16: (0, 2), 12: (4, 13), 11: (1, 1)} # m51e_best test_relabel_11_20
# d = {14: (47, 79), 13: (1382, 1457), 7: (428, 432), 6: (598, 606), 9: (67, 67), 10: (33, 33), 15: (3, 5), 16: (0, 2), 12: (2, 16), 8: (19, 22)} # m51e_best test_set_len26
d = {14: (42, 79), 13: (1383, 1457), 7: (431, 432), 6: (598, 606), 9: (67, 67), 10: (33, 33), 15: (2, 5), 16: (0, 2), 12: (1, 16), 8: (20, 22)} # m54e_best test_set_len26
```

```python
total = 0
true = 0
for k, (t1, t2) in d.items():
    true += t1
    total += t2
print(true, total, total - true, true / total)
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr'
max_text_len = 26
w, h = 100, 32
n_classes = 10
df_name = f'test_set_len{max_text_len}'
# df_name = f'test_relabel_11_20_len{max_text_len}'
model_num = 51
epoch = find_best_model_epoch(model_num)
batch_size = 20
img_dir = f'images_train_padding_size{w}x{h}'
grcl_fsize = 3
# grcl_niter = 3
grcl_niter = 3
lstm_units = 512
accuracy_threshold = 98.5
```

```python
metrics.test_metric_by_markets(dataset_dir_path, df_name, img_dir, model_num, epoch,
                               batch_size, max_text_len, h, w, n_classes,
                               grcl_niter, grcl_fsize, lstm_units, accuracy_threshold)
```

```python
test_d = {
    14: 0.038,
    13: 0.883,
    7: 0.958,
    6: 0.922,
    9: 1.0,
    10: 1.0,
    15: 0.0,
    16: 0.0,
    12: 0.188,
    8: 0.909,
}
train_d = {
    7: 0.918,
    9: 0.986,
    6: 0.926,
    8: 0.875,
    10: 0.869,
    11: 0.29,
    13: 0.934,
    12: 0.252,
    5: 0.858,
    14: 0.243,
    15: 0.601,
    16: 0.516,
    17: 0.333,
}


def print_metrics(test_d, train_d):
    for k in train_d.keys():
        if k in test_d:
            print(k, test_d[k], train_d[k])
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr'
df_name = 'test_set_google_bs_14'
metrics.test_google(dataset_dir_path, df_name)
```

```python
d = {14: (51, 79), 13: (828, 1457), 7: (407, 432), 6: (495, 606), 9: (58, 67), 10: (31, 33), 15: (3, 5), 16: (2, 2), 12: (3, 16), 8: (13, 22)}
```

```python
total = 0
true = 0
for k, (t1, t2) in d.items():
    true += t1
    total += t2
print(true, total, total - true, true / total)
```

```python

```
