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
import tensorflow as tf
from ilio import fread
from pathlib import Path
from ssd_mobilenet import metrics
```

```python
# model_path = '/srv/data_science/training/checkpoints/mobile_detection/model2/pretrain_model2/tag_detector.tflite'
model_path = '/srv/data_science/training/checkpoints/mobile_detection/model3/pretrain_model1/tag_detector_optimized.tflite'

base_dir = Path('/srv/data_science/storage/mobile_detection')

dataset_dir_path = base_dir / "dataframes"

img_dir = base_dir / 'imgs'

df_name = 'val_set1'

resize_shape = (320, 320)
iou_threshold = .4
num_filter = lambda x: x >= 2
debug = 0
n = 1000
```

```python
df_name = 'train'
base_dir = Path('/opt/data_sets/detection')
img_dir = base_dir / 'processed_images'
dataset_dir_path = base_dir / 'training'
img_format = 'png'
num_filter = lambda x: x <= 2
metrics.run_model(dataset_dir_path, df_name, model_path, img_dir, resize_shape, iou_threshold, num_filter, from_polygon=True, img_format=img_format)
```

```python
df_name = 'test_set1'
metrics.run_model(dataset_dir_path, df_name, model_path, img_dir, resize_shape, iou_threshold, debug=debug)
```

```python
df_name = 'train_set1'
metrics.run_model(dataset_dir_path, df_name, model_path, img_dir, resize_shape, iou_threshold)
```

```python
df_name = 'val_set1'
metrics.run_model(dataset_dir_path, df_name, model_path, img_dir, resize_shape, iou_threshold, debug=debug)
```

```python
!saved_model_cli show --dir $model_dir --signature_def serving_default 2> /dev/null
```

```python

```

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

```python
path = %env PYTHONPATH
%set_env PYTHONPATH=$path:ssd_mobilenet:ssd_mobilenet/slim
```

```python
model_dir = '/srv/data_science/training/checkpoints/mobile_detection/model2/pretrain_model2/inf_graph.pb/saved_model.pb'
config = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/ssd_mobilenet/configs/ssdlite_mobilenet_v3_large_1.config'
eval_dir = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/eval_results'
```

```python
!python ssd_mobilenet/object_detection/model_main.py --model_dir $model_dir --pipeline_config_path $config --checkpoint_dir $model_dir > output2.txt 2> errors2.txt
```

```python
!python ssd_mobilenet/object_detection/legacy/eval.py --eval_dir $eval_dir --pipeline_config_path $config --checkpoint_dir $model_dir > output2.txt 2> errors2.txt
```

```python

```
