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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

```python
path = %env PYTHONPATH
%set_env PYTHONPATH=$path:ssd_mobilenet:ssd_mobilenet/slim
```

```python
model_dir = '/srv/data_science/training/checkpoints/mobile_detection/model2/pretrain_model2'
model_dir_p = Path(model_dir)
trained_checkpoint = str(model_dir_p / 'model.ckpt-397722')
# output_directory = str(model_dir_p / 'inf_graph_noopt.pb')
output_directory = str(model_dir_p / 'inf_graph.pb')
config = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/ssd_mobilenet/configs/ssdlite_mobilenet_v3_large_1.config'
```

```python
model_dir = '/srv/data_science/training/checkpoints/mobile_detection/model3/pretrain_model1'
model_dir_p = Path(model_dir)
trained_checkpoint = str(model_dir_p / 'model.ckpt-399296')
output_directory = str(model_dir_p / 'inf_graph.pb')
config = '/srv/data_science/training/code/mobile_detection/ssd_mobilenet/ssd_mobilenet/configs/ssdlite_mobilenet_v3_large_2.config'
```

```python
!python ssd_mobilenet/object_detection/export_tflite_ssd_graph.py --pipeline_config_path $config --trained_checkpoint_prefix $trained_checkpoint --output_directory $output_directory --nooptimize
```

```python
!python ssd_mobilenet/object_detection/export_tflite_ssd_graph.py --helpfull
```

```python
# mobilenet_saved_model = str(model_dir_p / 'inf_graph179874.pb/saved_model.pb')
mobilenet_saved_model = str(model_dir_p / 'inf_graph.pb/saved_model.pb')
# mobilenet_saved_model = str(model_dir_p / 'inf_graph_noopt.pb/saved_model.pb')
output_file = str(model_dir_p / 'tag_detector_optimized.tflite')
# output_file = str(model_dir_p / 'tag_detector_quantized.tflite')

input_shapes="1,320,320,3"
output_arrays = "TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3"
input_arrays = "normalized_input_image_tensor"
inference_type = 'FLOAT'
# inference_type = 'QUANTIZED_UINT8'
# std_dev, mean = 128, 128
```

```python
!tflite_convert --graph_def_file=$mobilenet_saved_model --output_file=$output_file --input_arrays=$input_arrays --output_arrays=$output_arrays --input_shapes=$input_shapes --allow_custom_ops --inference_type $inference_type
```

```python
!tflite_convert --help
```

```python

```
