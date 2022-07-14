import cv2
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from math import ceil
from tools.utils import enableGPU, allow_growth
enableGPU(0)
allow_growth()
# disableGPU()


from keras_loss_function.keras_ssd_loss import SSDLoss
# from models.keras_ssd7 import build_model
# from models.keras_ssd7_1pred_layer import build_model
# from models.keras_ssd7_2pred_layers import build_model
# from models.pricetags.keras_ssd7_3pred_layers import build_model
from models.pricetags.keras_ssd7_3pred_layers_large_input import build_model
# from models.keras_ssd7_3pred_layers_cnn_mod import build_model
# from models.keras_tiny_ssd7 import build_model

from keras.utils import Sequence
from models.mobilenet_v2 import *
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_geometric_ops import ResizeRandomInterp, AlignMode

from data_generator.simple_generator import SimpleGenerator
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize

img_height = 60*2 # Height of the input images
img_width = 100*2 # Width of the input images
# scales = None
# scales = [0.08, 0.2, 0.5, 0.9, 0.96]  # ssd7
# aspect_ratios = [.2, .5, 1., 3.]
# aspect_ratios = [0.3, 1.0, 2.0, 3.0]
# aspect_ratios = [0.3, 1.0, 3.0]
#aspect_ratios = [0.33333, 0.5, 1.0, 2.0, 3.0] # The list of aspect ratios for the anchor boxes


# scales = [0.13, 0.28, 0.66, 0.85, 0.9]  # ssd7
# 50
# aspect_ratios_per_layer = [[.3, .6, 1., 1.5],
#                            [.3, .5, .9, 1.5],
#                            [.2, .6, .9, 2.0],
#                            [.3, .6, 1., 2.0]]

# 52
scales = [0.12, 0.25, 0.55, 0.85, 0.9]
aspect_ratios = [0.2, 0.45, 0.65, 0.85]


# # 42,43
# # scales = [0.01, 0.05, 0.15, 0.3, 0.9]  # ssd7
# # aspect_ratios = [0.2, 0.4, 0.6, 0.8]
#
#
# # 44, 45
# scales = [0.15, 0.25, 0.35, 0.55, 0.9]
# # scales = [0.13, 0.3, 0.66, 0.8, 0.9]
# aspect_ratios = [0.25, 0.45, 0.6, 0.8]
clip_boxes = True
same_ar = False
count_channels = 1

# conv_type = None
# conv_type = 'std'
# conv_type = 'fac'
conv_type = 'sep'
# conv_type = 'cp' # todo: проверить почему не работает

print('scales', scales,
      'aspect_ratios', aspect_ratios,
      # 'aspect_ratios_per_layer', aspect_ratios_per_layer,
      'clip_boxes', clip_boxes,
      'same_ar', same_ar,
      'count_channels', count_channels,
      'conv_type', conv_type)

two_boxes_for_ar1 = False # Whether or not you want to generate two anchor boxes for aspect ratio 1

steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
# clip_boxes = False  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries # 20, 21, 22
variances = [1., 1., 1., 1.] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
n_classes = 11

K.clear_session()

#model = MobileNetV2(image_size = (img_height, img_width, 3),include_top = False,pooling = 'avg')
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

#model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(img_height, img_width, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights=None, input_tensor=None, pooling=None, classes=1000)
#from keras.utils import plot_model
#plot_model(model, to_file='model1.png',show_shapes=True)

model, predictor_sizes = build_model(image_size=(img_height, img_width, count_channels),
                                       n_classes=n_classes,
                                       mode='training',
                                       l2_regularization=0.0005,
                                       scales=scales,
                                       aspect_ratios_global=aspect_ratios,
                                       # aspect_ratios_per_layer=aspect_ratios_per_layer,
                                       two_boxes_for_ar1=two_boxes_for_ar1,
                                       steps=steps,
                                       offsets=offsets,
                                       clip_boxes=clip_boxes,
                                       variances=variances,
                                       normalize_coords=normalize_coords,
                                       subtract_mean=127.5,
                                       divide_by_stddev=127.5,
                                       return_predictor_sizes = True,
                                       same_ar=same_ar,
                                       conv_type=conv_type)


# model.load_weights('2_1_ssd7_epoch-64_loss-0.7118_val_loss-0.7611.h5', by_name = True)
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                               'compute_loss': SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss})

optimizer = Adam()
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=optimizer, loss=SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss)


# model_path = 'digits1_ssd7_detection_epoch-163_loss-0.3716_val_loss-0.7662.h5'
# model_path = '/home/oleynik/project/digitsdetector_120x200_sep_depth_multiplier_2.h5'
# model = keras.models.load_model(model_path,
#                         custom_objects={'AnchorBoxes': AnchorBoxes,
#                                         'compute_loss': SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss})
# if 1==1:
#     print(model.summary())
#     exit()

#from keras.utils import plot_model
#plot_model(model, to_file='model3.png', show_shapes=True)

model_checkpoint = ModelCheckpoint(filepath='digits_detection_200x120_3_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto',
                                   period=1)
# model_checkpoint1 = ModelCheckpoint(filepath='digits_detection_200x120_1_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
#                                   monitor='loss',
#                                   verbose=1,
#                                   save_best_only=True,
#                                   mode='auto',
#                                   period=1)
#early_stopping = EarlyStopping(monitor='val_loss',
#                               min_delta=0.0,
#                               patience=10,
#                               verbose=1)
#

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.5,
                                         patience=30,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=10,
                                         min_lr=0.0000001)

print(model.summary())
print(predictor_sizes)
# model.save('model_1000x1000.h5')

# if True:
#     savepath = f'/home/oleynik/project/digitsdetector_120x200_{conv_type}_depth_multiplier_2.h5'
#     model.save(savepath)
#     print(savepath)
#     exit()

class Generator(Sequence):

    def __init__(self, batch_size, transformations, label_encoder, images, rects, random = False):
        self.rects = rects
        self.images = images
        self.batch_size = batch_size
        self.transformations = transformations
        self.label_encoder = label_encoder
        self.random = random
        print("Generator init")

    def __len__(self):
        if self.random:
            return 25
        else:
            return ceil(self.images.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        if self.random:
            np.random.seed()
            offset = 0
            r = np.random.choice(self.images.shape[0], self.batch_size, replace = False)
        else:
            offset = idx * self.batch_size
            r = range(self.batch_size)

        for i in r:
            if offset + i > self.images.shape[0] - 1:
                break

            batch_x.append(np.copy(self.images[offset + i]))
            batch_y.append(np.copy(self.rects[offset + i]))
            if self.transformations:
                for transform in self.transformations:
                    batch_x[-1], batch_y[-1] = transform(batch_x[-1], batch_y[-1])

        if self.label_encoder:
            batch_y = self.label_encoder(batch_y)

        batch_x = np.array(batch_x)
        return [batch_x, batch_y]

ssd_input_encoder = SSDInputEncoder(img_height = img_height,
                                    img_width = img_width,
                                    n_classes = n_classes,
                                    predictor_sizes = predictor_sizes,
                                    scales = scales,
                                    aspect_ratios_global = aspect_ratios,
                                    # aspect_ratios_per_layer=aspect_ratios_per_layer,
                                    two_boxes_for_ar1 = two_boxes_for_ar1,
                                    steps = steps, offsets = offsets,
                                    clip_boxes = clip_boxes,
                                    variances = variances,
                                    matching_type = 'multi',
                                    pos_iou_threshold = 0.5,
                                    neg_iou_limit = 0.3,
                                    normalize_coords = normalize_coords,
                                    same_ar=same_ar)

batch_size = 32

# dataset_path = '/home/ml/datasets/mars/price_tags/{}.npy'
# dataset_path = '/home/ml/datasets/schwarzkopf-retail/price_tags/{}.npy'

# train_pricetags_detector_19.txt только шварц
# train_pricetags_detector_20.txt (21,22) перепроверка точности марса с 15 моделью
# train_x = np.load(dataset_path.format('train_x'))
# train_y = np.load(dataset_path.format('train_y'))
# val_x = np.load(dataset_path.format('val_x'))
# val_y = np.load(dataset_path.format('val_y'))

# train_pricetags_detector_18.txt марс+шварц+аугментация
# 25 SimpleGenerator augmentator_variable_size
# train_x = np.load(dataset_path.format('train_x_mars_schwarzkopf'))
# train_y = np.load(dataset_path.format('train_y_mars_schwarzkopf'))
# val_x = np.load(dataset_path.format('val_x_mars_schwarzkopf'))
# val_y = np.load(dataset_path.format('val_y_mars_schwarzkopf'))

# dataset_path = '/home/ml/datasets/schwarzkopf-retail/price_tags'
# train_x = np.load(f'{dataset_path}/train_x_mars_schwarzkopf_14282.npy')
# train_y = np.load(f'{dataset_path}/train_y_mars_schwarzkopf_14282.npy')
# val_x = np.load(f'{dataset_path}/val_x_mars_schwarzkopf_14282.npy')
# val_y = np.load(f'{dataset_path}/val_y_mars_schwarzkopf_14282.npy')

dataset_path = '/home/ml/datasets/schwarzkopf-retail/price_tags'
train_x = np.load(f'{dataset_path}/train_x_mars_schwarzkopf_16577.npy')
train_y = np.load(f'{dataset_path}/train_y_mars_schwarzkopf_16577.npy')
train_filenames = np.load(f'{dataset_path}/train_filenames_mars_schwarzkopf_16577.npy')
val_x = np.load(f'{dataset_path}/val_x_mars_schwarzkopf_16577.npy')
val_y = np.load(f'{dataset_path}/val_y_mars_schwarzkopf_16577.npy')
val_filenames = np.load(f'{dataset_path}/val_filenames_mars_schwarzkopf_16577.npy')

# filter schwarzkopf
indexes = np.flatnonzero(['schwarzkopf-retail/price_tags' in i for i in train_filenames])
train_x = train_x[indexes]
train_y = train_y[indexes]
train_filenames = train_filenames[indexes]
indexes = np.flatnonzero(['schwarzkopf-retail/price_tags' in i for i in val_filenames])
val_x = val_x[indexes]
val_y = val_y[indexes]
val_filenames = val_filenames[indexes]
print(len(train_x), len(val_x), len(train_x)+len(val_x))


# # filter mars
# indexes = np.flatnonzero(['mars/price_tags' in i for i in train_filenames])
# train_x = train_x[indexes]
# train_y = train_y[indexes]
# train_filenames = train_filenames[indexes]
# indexes = np.flatnonzero(['mars/price_tags' in i for i in val_filenames])
# val_x = val_x[indexes]
# val_y = val_y[indexes]
# val_filenames = val_filenames[indexes]
# print(len(train_x), len(val_x), len(train_x)+len(val_x))



# def filter_percent_data(y):
#     filtered_indexes = []
#     i = 0
#     for signs in y:
#         j = 0
#         for sign_class_id, xmin, ymin, xmax, ymax in signs:
#             if sign_class_id == 11:
#                 j += 1
#         if j > 1:
#             pass
#         else:
#             filtered_indexes.append(i)
#         i += 1
#     return filtered_indexes
#
# train_indexes = filter_percent_data(train_y)
# val_indexes = filter_percent_data(val_y)
# train_x = train_x[train_indexes]
# train_y = train_y[train_indexes]
# val_x = val_x[val_indexes]
# val_y = val_y[val_indexes]

# 23 - обычная модель на маленьких + augmentator_variable_size SimpleGenerator
# 24 - модель с настроенными scale и aspect_ratios + augmentator_variable_size SimpleGenerator
# train_x = np.load(dataset_path.format('train_x_mars_schwarzkopf_small_digits'))
# train_y = np.load(dataset_path.format('train_y_mars_schwarzkopf_small_digits'))
# val_x = np.load(dataset_path.format('val_x_mars_schwarzkopf_small_digits'))
# val_y = np.load(dataset_path.format('val_y_mars_schwarzkopf_small_digits'))


#
# train_x = np.load(dataset_path.format('train_x_small_digits'))
# train_y = np.load(dataset_path.format('train_y_small_digits'))
# val_x = np.load(dataset_path.format('val_x_small_digits'))
# val_y = np.load(dataset_path.format('val_y_small_digits'))

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
resizing = Resize(img_height,
                  img_width,
                  interpolation_mode='custom',
                  box_filter=box_filter_resize,
                  keep_original_aspect_ratio=True,
                  align_mode=AlignMode.CENTER)

# train_generator = Generator(batch_size, [resizing, augmentator_constant_size], ssd_input_encoder, train_x, train_y, False)
# val_generator = Generator(len(val_x), [resizing], ssd_input_encoder, val_x, val_y, False)
# val_generator = Generator(batch_size, [resizing], ssd_input_encoder, val_x, val_y, False)

# # BalancedBatchGenerator
# train_p, bins = calc_hist(train_x,train_y)
# train_p[6]+=sum(train_p[7:])
# train_p = train_p[:7]
# train_inv_p = calc_inverse_probability(train_p)
#
# train_balanced_generator = BalancedBatchGeneratorForDetection(train_x,
#                                                               train_y,
#                                                               [augmentator_variable_size], # 21 23 24
#                                                               # [augmentator_constant_size, rand_interp_resizing],  #22
#                                                               ssd_input_encoder,
#                                                               split_points=[0,10,20,30,40,50,60,100],
#                                                               probabilities=train_inv_p,
#                                                               batch_size=batch_size,
#                                                               verbose=False)

variable_train_generator = SimpleGenerator(train_x,
                                  train_y,
                                  [augmentator_variable_size]+([convert_gray] if count_channels == 1 else []),
                                  ssd_input_encoder,
                                  batch_size,
                                  random=False)
constant_train_generator = SimpleGenerator(train_x,
                                  train_y,
                                  [augmentator_constant_size, rand_interp_resizing]+([convert_gray] if count_channels == 1 else []),
                                  ssd_input_encoder,
                                  batch_size,
                                  random=False)

# train_generator = ClassesAugmGenerator(train_x,
#                                   train_y,
#                                   [augmentator_variable_size],
#                                   # [augmentator_constant_size, rand_interp_resizing],
#                                   ssd_input_encoder,
#                                   batch_size)

val_generator = SimpleGenerator(val_x, val_y, [resizing]+([convert_gray] if count_channels == 1 else []), ssd_input_encoder, batch_size, random=False)


# save generated images
def save_images_from_generator(generator, x, y, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        for f_name in os.listdir(save_path):
            os.remove(os.path.join(save_path,f_name))
    img_count = 0
    for batch_id in range(len(generator)):
        batch_x = generator[batch_id][0]
        # batch_y = generator[batch_id][1]
        for idx in range(len(batch_x)):
            img_path = os.path.join(save_path, '{:010d}.png'.format(img_count))
            print('file: {} marks: {}\n{}\n'.format(img_path, len(y[img_count]), y[img_count]))
            cv2.imwrite(img_path, batch_x[idx])

            img_path = os.path.join(save_path, '{:010d}_original.png'.format(img_count))
            cv2.imwrite(img_path, x[img_count])
            img_count += 1

# save_images_from_generator(train_generator, train_x, train_y, '/home/oleynik/project/train_gen_imgs')
# save_images_from_generator(val_generator, val_x, val_y, '/home/oleynik/project/val_gen_imgs')


model.fit_generator(
                    # generator=train_balanced_generator,
                    generator=variable_train_generator,
                    # generator=constant_train_generator,
                    validation_data = val_generator,
                    callbacks=[model_checkpoint,
                               # model_checkpoint1,
                               reduce_learning_rate
                               ],
                    epochs = 10000,
                    shuffle = True,
                    use_multiprocessing = True,
                    workers = 32,
                    # steps_per_epoch = 350,
                    max_queue_size = 100
                    )

print("Done")
