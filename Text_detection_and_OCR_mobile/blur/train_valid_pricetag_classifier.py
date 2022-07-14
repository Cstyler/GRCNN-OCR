from tools.utils import enableGPU, allow_growth, disableGPU
enableGPU(2)
allow_growth()
# disableGPU()

import mlflow
import os
import glob
from sklearn.model_selection import train_test_split
from models.digits_classifier import digits_classifier_model
import tensorflow
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from keras_callbacks.mlflow_model_callbacks import MLFlowModelCheckpointCallback
from tensorflow_core.python.keras.utils.np_utils import to_categorical


from multiprocessing.pool import Pool
from data_generator.object_detection_2d_geometric_ops import RandomTranslate, ResizeRandomInterp
from data_generator.triplet_generators import KNDataGenerator
# from sku_classifier.calc_acc_sku_classifier import DataAugmentationClassifier
# from sku_classifier.calc_acc_sku_classifier import DataAugmentationClassifier
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationClassifier
from data_generator.object_detection_2d_photometric_ops import ConvertColor
import tensorflow as tf
if tf.__version__[0] == "2":
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
    from tensorflow.python.keras.models import load_model
else:
    from keras.optimizers import Adam
    from keras.callbacks import ReduceLROnPlateau, Callback
from data_generator.object_detection_2d_photometric_ops import *
from models.classifier_triplet_mars import classifier, classifier_cross_entropy
from tools.image_utils import resize_image
from SkynetCV import SkynetCV


def resize_lambda(f):
    try:
        return resize_image(f, height=img_height, width=img_width)
    except:
        print('f', f)


# img_height = 60
# img_width = 60
n_classes = 2

img_height = 32
img_width = 32

model = digits_classifier_model(image_size=(img_height, img_width, 1),
                                classes=n_classes,
                                subtract_mean=127.5,
                                divide_by_stddev=127.5)

# model = classifier_cross_entropy(image_size=(img_height, img_width, 1),
#                                  n_classes=n_classes,
#                                  subtract_mean=127.5,
#                                  divide_by_stddev=127.5)

print(model.summary())


model.compile(optimizer=Adam(learning_rate=0.001),
              loss=tensorflow.keras.losses.categorical_crossentropy,
              metrics=['acc'])

print('model.metrics', model.metrics)
print('model.metrics_names', model.metrics_names)

#filepath = '/home/oleynik/saved_models/blur_classifier_5_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}_val_acc-{val_acc:.4f}.h5'

reduce_learning_rate = ReduceLROnPlateau(monitor='val_acc',
                                         factor=0.75,
                                         patience=20,
                                         verbose=1,
                                         cooldown=10)


# root_dataset_dir = '/home/ml/datasets/price_tags/validity_dataset_992'
root_dataset_dir = '/home/ml/datasets/price_tags/validity_dataset_5235/new/'
train_filenames_path = os.path.join(root_dataset_dir, 'train_filenames2.npy')
train_annotations_path = os.path.join(root_dataset_dir, 'train_annotations2.npy')
val_filenames_path = os.path.join(root_dataset_dir, 'val_filenames2.npy')
val_annotations_path = os.path.join(root_dataset_dir, 'val_annotations2.npy')

# # ----- start create datasets -----
# good_paths = glob.glob(os.path.join(root_dataset_dir, 'good/*'))
# bad_paths = glob.glob(os.path.join(root_dataset_dir, 'bad/*'))
#
# good_train_filenames, good_val_filenames = train_test_split(good_paths,
#                                                      test_size=0.2,
#                                                      random_state=42)
# bad_train_filenames, bad_val_filenames = train_test_split(bad_paths,
#                                                    test_size=0.2,
#                                                    random_state=42)
#
# train_filenames = np.concatenate([good_train_filenames, bad_train_filenames])
# train_annotations = np.concatenate([np.ones(len(good_train_filenames)), np.zeros(len(bad_train_filenames))])
# val_filenames = np.concatenate([good_val_filenames, bad_val_filenames])
# val_annotations = np.concatenate([np.ones(len(good_val_filenames)), np.zeros(len(bad_val_filenames))])
#
# np.save(train_filenames_path, train_filenames)
# np.save(train_annotations_path, train_annotations)
# np.save(val_filenames_path, val_filenames)
# np.save(val_annotations_path, val_annotations)
# # ----- end create datasets -----


train_filenames = np.load(train_filenames_path)
train_annotations = np.load(train_annotations_path)
val_filenames = np.load(val_filenames_path)
val_annotations = np.load(val_annotations_path)


mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("danone_validpricetagclassifier")
mlflow.start_run(run_name='SKYNET-273')
mlflow.log_param('train_filenames', train_filenames_path)
mlflow.log_param('train_annotations', train_annotations_path)
mlflow.log_param('val_filenames', val_filenames_path)
mlflow.log_param('val_annotations', val_annotations_path)
mlflow.log_param('img_hw', (img_height, img_width))
#mlflow.log_artifact('/home/oleynik/saved_models/blur_classifier_5_epoch-361_loss-0.2574_val_loss-0.2324_val_acc-0.9189.h5')


val_filenames = [SkynetCV.load(i) for i in val_filenames]
val_filenames = [ConvertColor('BGR', 'GRAY', gray_channels_count=1)(i) for i in val_filenames]
val_filenames = np.array([resize_lambda(i) for i in val_filenames])


train_gen = KNDataGenerator(train_filenames,
                            train_annotations,
                            transformations=[DataAugmentationClassifier(img_height, img_width,
                                                                        # random_brightness=(-12, 12, 0.5),
                                                                        random_brightness=(-48, 48, 0.5),
                                                                        # random_contrast=(0.9, 1.7, 0.5),
                                                                        random_contrast=(0.5, 1.8, 0.5),
                                                                        # random_saturation=(0.9, 1.1, 0.5),
                                                                        random_saturation=(0.5, 1.8, 0.5),
                                                                        random_hue=(4, 0.5),
                                                                        random_flip=0.5,
                                                                        # random_flip=0.0,
                                                                        # random_translate=((0.03, 0.3), (0.03, 0.3), 0.5),
                                                                        random_translate=((0.03, 0.2), (0.03, 0.2), 0.5),
                                                                        random_scale=(0.9, 1.1, 0.5),
                                                                        background=(0, 0, 0),
                                                                        # random_stretch=((0.8, 1.2), (0.8, 1.2), 0.5),
                                                                        random_stretch=((0.9, 1.1), (0.9, 1.1), 0.5),
                                                                        random_rotate=([180], 0.5),
                                                                        random_rotate_angle=(-5, 5, .5)
                                                                        # background_source=train_x_all
                                                                        ),
                                             ConvertColor('BGR', 'GRAY', gray_channels_count=1)],
                            k=n_classes,
                            n=1,
                            add_categorical=True)

# model_checkpoint = ModelCheckpoint(
#                                    monitor='val_acc',
#                                    verbose=1,
#                                    save_best_only=True,
#                                    save_weights_only=False,
#                                    mode='auto')

model_checkpoint = MLFlowModelCheckpointCallback(
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='auto')

model.fit_generator(generator=train_gen.generate(),
                    callbacks=[model_checkpoint, reduce_learning_rate],
                    validation_data=(val_filenames, to_categorical(val_annotations, n_classes)),
                    workers=os.cpu_count(),
                    use_multiprocessing=True,
                    epochs=1000,
                    shuffle=False,
                    steps_per_epoch=100)

print("Done")
