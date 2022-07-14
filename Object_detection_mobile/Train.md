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
from pathlib import Path
```

```python
path = %env PYTHONPATH
%set_env PYTHONPATH=$path:ssd_mobilenet:ssd_mobilenet/slim
```

```python
# model_dir = '/srv/data_science/training/checkpoints/mobile_detection/model1'
model_dir = '/srv/data_science/training/checkpoints/mobile_detection/model2'

config = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/ssd_mobilenet/configs/ssdlite_mobilenet_v3_large_1.config'
```

```python
model_dir = '/srv/data_science/training/checkpoints/mobile_detection/model3'
config = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/ssd_mobilenet/configs/ssdlite_mobilenet_v3_large_2.config'
```

```python
model_dir = '/srv/data_science/training/checkpoints/mobile_detection/model4'
config = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/ssd_mobilenet/configs/ssdlite_mobilenet_v3_large_3.config'
# config = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/ssd_mobilenet/configs/ssdlite_mobilenet_v3_large_2.config'
```

```python
!python ssd_mobilenet/object_detection/model_main.py --model_dir $model_dir --pipeline_config_path $config > output.txt 2> errors.txt
```

```python
!python ssd_mobilenet/object_detection/legacy/train.py --logtostderr --train_dir $model_dir --pipeline_config_path $config > output.txt 2> errors.txt
```

```python

```
