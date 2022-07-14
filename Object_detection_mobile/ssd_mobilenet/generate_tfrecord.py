import io
import sys
from pathlib import Path

import tensorflow as tf
import tqdm
from PIL import Image

from pylibs import pandas_utils
from pylibs.json_utils import read_json_str
from pylibs.pandas_utils import DF_FILE_FORMAT
from pylibs.rect_utils import UniversalRect
from pylibs.storage_utils import get_file_sharding

sys.path.append("ssd_mobilenet")

from object_detection.utils import dataset_util


def create_tf_example(img_dir, photo_id, row):
    path = str(get_file_sharding(img_dir, photo_id))
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()

    img = Image.open(io.BytesIO(encoded_jpg))
    assert img.format == 'JPEG'
    photo_w, photo_h = img.size

    filename = str(path).encode('utf8')
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    tags = read_json_str(row['tags'])['tags']
    for tag in tags:
        rect = UniversalRect.from_coords_dict(tag)
        x, y, w, h = rect.xywh
        half_w, half_h = w // 2, h // 2
        x_min = x - half_w
        x_max = x + half_w
        y_min = y - half_h
        y_max = y + half_h
        xmins.append(x_min / photo_w)
        xmaxs.append(x_max / photo_w)
        ymins.append(y_min / photo_h)
        ymaxs.append(y_max / photo_h)
        classes_text.append(b"tag")
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height'            : dataset_util.int64_feature(photo_h),
            'image/width'             : dataset_util.int64_feature(photo_w),
            'image/filename'          : dataset_util.bytes_feature(filename),
            'image/source_id'         : dataset_util.bytes_feature(filename),
            'image/encoded'           : dataset_util.bytes_feature(encoded_jpg),
            'image/format'            : dataset_util.bytes_feature('jpeg'.encode('utf-8')),
            'image/object/bbox/xmin'  : dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax'  : dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin'  : dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax'  : dataset_util.float_list_feature(ymaxs),
            'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def generate_tfrecord(dataset_dir_path, df_name, img_dir):
    dataset_dir_path = Path(dataset_dir_path)
    tf_record_path = str(dataset_dir_path / f'{df_name}.record')
    writer = tf.python_io.TFRecordWriter(tf_record_path)

    df_path = dataset_dir_path / (DF_FILE_FORMAT % df_name)
    df = pandas_utils.read_dataframe(df_path)
    for photo_id, row in tqdm.tqdm_notebook(df.iterrows(), total=len(df.index)):
        tf_example = create_tf_example(img_dir, photo_id, row)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(tf_record_path))


if __name__ == '__main__':
    tf.app.run()
