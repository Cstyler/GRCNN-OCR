import cv2
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from math import ceil
from tools.utils import enableGPU, allow_growth
enableGPU(1)
allow_growth()
# disableGPU()


from keras_loss_function.keras_ssd_loss import SSDLoss
from models.pricetags.keras_ssd7_3pred_layers import build_model
from models.mobilenetv3.ssd_mobilenet_v3_small_pricetags import ssd_mobilenet_v3_small_pricetags

from tensorflow.keras.utils import Sequence
from models.mobilenet_v2 import *
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_geometric_ops import ResizeRandomInterp, AlignMode

from data_generator.simple_generator import SimpleGenerator
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

img_height = 60 # Height of the input images
img_width = 100 # Width of the input images
# scales = None
# scales = [0.08, 0.2, 0.5, 0.9, 0.96]  # ssd7
# aspect_ratios = [.2, .5, 1., 3.]
# aspect_ratios = [0.3, 1.0, 2.0, 3.0]
# aspect_ratios = [0.3, 1.0, 3.0]
#aspect_ratios = [0.33333, 0.5, 1.0, 2.0, 3.0] # The list of aspect ratios for the anchor boxes


# 50
# scales = [0.13, 0.28, 0.66, 0.85, 0.9]  # ssd7
# aspect_ratios_global = None
# aspect_ratios_per_layer = [[.3, .6, 1., 1.5],
#                            [.3, .5, .9, 1.5],
#                            [.2, .6, .9, 2.0],
#                            [.3, .6, 1., 2.0]]

min_scale = 0.05
max_scale = 0.95

scales = [
    1 / 38. * 2,
    1 / 19. * 2,
    1 / 10. * 2, 1 / 5. * 2, 1 / 3. * 2, 0.95, 1.0]
# aspect_ratios_global = [0.3333, 0.5, 1.0, 2.0, 3.0]
aspect_ratios_global = None
aspect_ratios_per_layer = [
    [1 / 3., 0.5, 1.0, 2.0, 3.0],
                           [0.25, 1 / 3., 0.5, 1.0, 2.0, 3.0, 4.0],
                           [0.25, 1 / 3., 0.5, 1.0, 2.0, 3.0, 4.0],
                           [0.25, 1 / 3., 0.5, 1.0, 2.0, 3.0, 4.0],
                           [0.25, 1 / 3., 0.5, 1.0, 2.0, 3.0, 4.0],
                           [0.25, 1 / 3., 0.5, 1.0, 2.0, 3.0, 4.0]]

# clip_boxes = False  # 50
clip_boxes = True
same_ar = False
count_channels = 3


print('scales', scales,
      # 'aspect_ratios', aspect_ratios,
      'aspect_ratios_per_layer', aspect_ratios_per_layer,
      'clip_boxes', clip_boxes,
      'same_ar', same_ar,
      'count_channels', count_channels)

two_boxes_for_ar1 = False # Whether or not you want to generate two anchor boxes for aspect ratio 1

steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
# clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries # 20, 21, 22
variances = [.1, .1, .2, .2] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
n_classes = 11

K.clear_session()

# model, predictor_sizes = build_model(image_size=(img_height, img_width, count_channels),
#                                      n_classes=n_classes,
#                                      min_scale=min_scale,
#                                      max_scale=max_scale,
#                                      l2_regularization=0.00004,
#                                      scales=scales,
#                                      aspect_ratios_global=aspect_ratios_global,
#                                      aspect_ratios_per_layer=aspect_ratios_per_layer,
#                                      two_boxes_for_ar1=two_boxes_for_ar1,
#                                      steps=steps,
#                                      offsets=offsets,
#                                      clip_boxes=clip_boxes,
#                                      variances=variances,
#                                      normalize_coords=normalize_coords,
#                                      subtract_mean=127.5,
#                                      divide_by_stddev=127.5,
#                                      return_predictor_sizes=True,
#                                      same_ar=same_ar,
#                                      # output_activation=None)
#                                      output_activation='softmax')

model, predictor_sizes = ssd_mobilenet_v3_small_pricetags(image_size=(img_height, img_width, count_channels),
                           n_classes=n_classes,
                           l2_regularization=0.00004,
                           min_scale=min_scale,
                           max_scale=max_scale,
                           scales=scales,
                           aspect_ratios_global=aspect_ratios_global,
                           aspect_ratios_per_layer=aspect_ratios_per_layer,
                           two_boxes_for_ar1=two_boxes_for_ar1,
                           steps=steps,
                           offsets=offsets,
                           clip_boxes=clip_boxes,
                           variances=variances,
                           normalize_coords=normalize_coords,
                           return_predictor_sizes=True,
                           same_ar=same_ar,
                           output_activation=None)

# # load trained model
# K.clear_session()
# # from tools.inference import load_keras_model
# from tensorflow.keras.models import load_model
# from models.mobilenetv3.ssd_mobilenet_v3_small import ssd_mobilenet_v3_small, _hard_swish, _relu6, _GlobalAveragePooling2D
# model_path = '/home/oleynik/project/saved_models/sku_detection_v3_500x500_mars_n_classes_1_BAT_1_epoch-419_loss-1.3960_val_loss-1.4115.h5'
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'compute_loss': SSDLoss(neg_pos_ratio=3, alpha=1.0, encode_background_as_zeros=True, classification_loss_type='sigmoid').compute_loss,
#                                                '_hard_swish': _hard_swish,
#                                                '_relu6': _relu6,
#                                                '_GlobalAveragePooling2D': _GlobalAveragePooling2D
#                                                })
# print(f'model_path: \'{model_path}\'')

lr = 0.5
print(f'learning_rate: {lr}')
model.compile(
              optimizer=Adam(learning_rate=lr),
              # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              loss=SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss
              # loss=SSDLoss(neg_pos_ratio=3, alpha=1.0, encode_background_as_zeros=True, classification_loss_type='sigmoid').compute_loss
              )

model_checkpoint = ModelCheckpoint(filepath='/home/oleynik/project/saved_models/new_digits_detection_100x60_1_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto',
                                   period=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.5,
                                         patience=20,
                                         cooldown=5,
                                         min_lr=0.0000001,
                                         verbose=1)

print(model.summary())
print(predictor_sizes)
# exit()

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    min_scale=min_scale,
                                    max_scale=max_scale,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios_global,
                                    aspect_ratios_per_layer=aspect_ratios_per_layer,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords,
                                    same_ar=same_ar,
                                    # encode_background_as_zeros=True)
                                    encode_background_as_zeros=False)

batch_size = 32

dataset_path = '/home/ml/datasets/schwarzkopf-retail/price_tags'
train_x = np.load(f'{dataset_path}/train_x_mars_schwarzkopf_16577.npy')
train_y = np.load(f'{dataset_path}/train_y_mars_schwarzkopf_16577.npy')
train_filenames = np.load(f'{dataset_path}/train_filenames_mars_schwarzkopf_16577.npy')
val_x = np.load(f'{dataset_path}/val_x_mars_schwarzkopf_16577.npy')
val_y = np.load(f'{dataset_path}/val_y_mars_schwarzkopf_16577.npy')
val_filenames = np.load(f'{dataset_path}/val_filenames_mars_schwarzkopf_16577.npy')

# # filter schwarzkopf
# indexes = np.flatnonzero(['schwarzkopf-retail/price_tags' in i for i in train_filenames])
# # indexes = np.flatnonzero(['mars/price_tags' in i for i in train_filenames])
# train_x = train_x[indexes]
# train_y = train_y[indexes]
# train_filenames = train_filenames[indexes]
# indexes = np.flatnonzero(['schwarzkopf-retail/price_tags' in i for i in val_filenames])
# # indexes = np.flatnonzero(['mars/price_tags' in i for i in val_filenames])
# val_x = val_x[indexes]
# val_y = val_y[indexes]
# val_filenames = val_filenames[indexes]
# print(len(train_x), len(val_x), len(train_x)+len(val_x))

from data_generator.data_augmentation_chain_variable_input_size import *
box_filter_resize = BoxFilter(check_overlap=False,
                                           check_min_area=True,
                                           check_degenerate=True,
                                           min_area=10)
convert_gray = ConvertColor(current='BGR', to='GRAY', keep_3ch=False, keep_1ch=True)

augmentator_variable_size = DataAugmentationVariableInputSize(
    img_height,
    img_width,
    random_brightness=(-20, 15, 0.4),
    random_contrast=(0.8, 1.2, 0.4),
    random_saturation=(0.8, 1.2, 0.4),
    random_hue=(18, 0.4),
    random_flip=0.,
    min_scale=0.8,
    max_scale=2.0,
    min_aspect_ratio=0.5,
    max_aspect_ratio=2.0,
    n_trials_max=20,
    clip_boxes=True,
    overlap_criterion='area',
    bounds_box_filter=(0.5, 1.0),
    bounds_validator=(0.7, 1.0),
    n_boxes_min=1,
    background=(0,0,0))

augmentator_constant_size = DataAugmentationConstantInputSize(
                                 random_brightness=(-48, 48, 0.5),
                                 # random_brightness=(-48, 48, 0.2),
                                 # random_brightness=(-10, 10, 0.3),
                                 random_contrast=(0.5, 1.8, 0.5),
                                 # random_contrast=(0.5, 1.8, 0.2),
                                 # random_contrast=(0.95, 1.05, 0.3),
                                 random_saturation=(0.5, 1.8, 0.5),
                                 # random_saturation=(0.5, 1.8, 0.2),
                                 # random_saturation=(0.95, 1.05, 0.3),
                                 random_hue=(18, 0.5),
                                 # random_hue=(18, 0.2),
                                 # random_hue=(1, 0.3),
                                 random_flip=0,
                                 # random_flip=0.2,
                                 # random_flip=0.,
                                 random_translate=((0.03, 0.5), (0.03, 0.5), 0.75),
                                 # random_translate=((0.03, 0.5), (0.03, 0.5), 1.),
                                 # random_translate=((0.03, 0.5), (0.03, 0.5), 0.75),
                                 random_scale=(0.5, 2.0, 0),
                                 random_stretch=((0.5, 2.), (0.5, 2.), 0.5),
                                 # random_stretch=((1., 2.), (1., 2.), 0.75),
                                 n_trials_max=3,
                                 clip_boxes=True,
                                 overlap_criterion='area',
                                 bounds_box_filter=(0.3, 1.0),
                                 bounds_validator=(0.5, 1.0),
                                 n_boxes_min=1,
                                 background=(0, 0, 0))

rand_interp_resizing = ResizeRandomInterp(img_height,
                                          img_width,
                                          box_filter=box_filter_resize,
                                          keep_original_aspect_ratio=True,
                                          align_mode=AlignMode.CENTER)
resizer = Resize(img_height,
                 img_width,
                 interpolation_mode='custom',
                 box_filter=box_filter_resize,
                 keep_original_aspect_ratio=True,
                 align_mode=AlignMode.CENTER)

val_images = []
for i in range(len(val_y)):
    im, ann = resizer(val_x[i], val_y[i])
    val_y[i] = ann
    val_images.append(im)
val_x = np.array(val_images)

# train_images = []
# for i in range(len(train_y)):
#     im, ann = resizer(train_x[i], train_y[i])
#     train_y[i] = ann
#     train_images.append(im)
# train_x = np.array(train_images)

augmentator = SSDDataAugmentation(img_height=img_height,
                                  img_width=img_width,
                                  photometric_distortions=True,
                                  expand=False,
                                  random_flip=False,
                                  random_crop=True,
                                  random_rotate=False)

train_generator = SimpleGenerator(images=train_x,
                                  annotations=train_y,
                                  transformations=[augmentator],
                                  label_encoder=ssd_input_encoder,
                                  batch_size=batch_size)


variable_train_generator = SimpleGenerator(train_x,
                                  train_y,
                                  [augmentator_variable_size]+([convert_gray] if count_channels == 1 else []),
                                  ssd_input_encoder,
                                  batch_size)
constant_train_generator = SimpleGenerator(train_x,
                                  train_y,
                                  [augmentator_constant_size, rand_interp_resizing]+([convert_gray] if count_channels == 1 else []),
                                  ssd_input_encoder,
                                  batch_size)

# model.fit(x=train_generator.generate(),
# model.fit(x=variable_train_generator.generate(),
model.fit(x=constant_train_generator.generate(),
          steps_per_epoch=int(len(train_x)/batch_size),
          # steps_per_epoch = 350,
          # initial_epoch=420,
          validation_data=(val_x, ssd_input_encoder(val_y)),
          callbacks=[model_checkpoint, reduce_learning_rate],
          epochs=10000,
          shuffle=True,
          use_multiprocessing=True,
          workers=32,
          max_queue_size=100
          )

print("Done")
