from multiprocessing.dummy import Pool
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import save_model
from SkynetCV.SkynetCV import AlignMode

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter
from keras_loss_function.keras_ssd_loss import *
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from metrics.metrics import calc_acc_f1
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from bounding_box_utils.bounding_box_utils import *
from models.mobilenetv3.ssd_mobilenet_v3_small import ssd_mobilenet_v3_small, _hard_swish, _relu6, _GlobalAveragePooling2D
import os
from tools.utils import disableGPU, enableGPU, allow_growth
from tools.draw_utils import plot_detector_accuracy
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from data_generator.object_detection_2d_geometric_ops import BoxFilter
from data_generator.object_detection_2d_photometric_ops import ConvertColor
from tools.inference import load_keras_model

disableGPU()
# enableGPU(0)
# allow_growth()


# model_filepath = '/home/ml/models/mars/digits1_ssd7_detection_25_epoch-622_loss-1.4638_val_loss-1.4696.h5'
# # acc_mean= 0.9118846183946483 acc_all= 0.9214163984093922 f1_mean= 0.9288585778108109 f1_all= 0.9591012121809402
# # confidence_thresh acc_mean= 0.35 acc_all= 0.95 f1_mean= 0.3 f1_all= 0.95
# # iou_thresh        acc_mean= 0.15 acc_all= 0.25 f1_mean= 0.2 f1_all= 0.25
# # after fix calc
# # acc_micro: 0.8986241853729182 acc_macro: 0.9386838427793108 f1_micro: 0.946605644546148 f1_macro: 0.9552244973121645
# # confidence_thresh acc_micro: 0.95 acc_macro: 0.95 f1_micro: 0.95 f1_macro: 0.95
# # iou_thresh        acc_micro: 0.2 acc_macro: 0.2 f1_micro: 0.2 f1_macro: 0.2
# data_path = '/home/ml/datasets/schwarzkopf-retail/price_tags/{}.npy'
# val_images = np.load(data_path.format('val_x_mars_schwarzkopf'))
# val_annotations = np.load(data_path.format('val_y_mars_schwarzkopf'))

model_filepath = '/home/ml/models/mars/digits1_ssd7_detection_50_epoch-483_loss-0.9535_val_loss-0.9578.h5'
# acc_mean= 0.9605778810169053 acc_all= 0.9491010973616624 f1_mean= 0.9752100306393213 f1_all= 0.9738859607091519
# confidence_thresh acc_mean= 0.9 acc_all= 0.85 f1_mean= 0.85 f1_all= 0.85
# iou_thresh        acc_mean= 0.2 acc_all= 0.2 f1_mean= 0.2 f1_all= 0.2
# after fix calc
# acc_micro: 0.9282911649303207 acc_macro: 0.9550906108733048 f1_micro: 0.9628122368790344 f1_macro: 0.9688632239598355
# confidence_thresh acc_micro: 0.95 acc_macro: 0.95 f1_micro: 0.95 f1_macro: 0.95
# iou_thresh        acc_micro: 0.15 acc_macro: 0.2 f1_micro: 0.15 f1_macro: 0.2
dataset_path = '/home/ml/datasets/schwarzkopf-retail/price_tags'
val_images = np.load(f'{dataset_path}/val_x_mars_schwarzkopf_14282.npy')
val_annotations = np.load(f'{dataset_path}/val_y_mars_schwarzkopf_14282.npy')
val_filenames = np.load(f'{dataset_path}/val_filenames_mars_schwarzkopf_14282.npy')

# model_filepath = '/home/oleynik/project/digits_detection_100x60_1_epoch-567_loss-1.3145_val_loss-1.5464.h5'
# acc_mean= 0.9307342987700132 acc_all= 0.9072063178677197 f1_mean= 0.9577064782135883 f1_all= 0.9513457556935817
# confidence_thresh acc_mean= 0.49999999999999994 acc_all= 0.49999999999999994 f1_mean= 0.49999999999999994 f1_all= 0.49999999999999994
# iou_thresh        acc_mean= 0.2 acc_all= 0.2 f1_mean= 0.2 f1_all= 0.2

print(f'model_filepath = \'{model_filepath}\'')

count_channels = 3

# linspace_num = 19
confidence_thresh_num = 19
iou_thresh_num = 19

# for list_classes in [[k] for k in range(1, 12)]:
# list_classes = [10]
classs = list(range(1, 12))


model_filename = os.path.basename(model_filepath)
res_filename = f'{model_filename}_val_{confidence_thresh_num}_{iou_thresh_num}_{classs if isinstance(classs, int) else "_".join([str(i)for i in classs])}.npy'


def validate():
    global val_images, val_annotations, val_filenames

    model = load_keras_model(model_filepath, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                             'compute_loss': SSDLoss(neg_pos_ratio=3,
                                                                                     alpha=1.0).compute_loss})

    input_shape = model.input_shape[1:]
    print(f'input_shape = {input_shape}')
    print(f'output_shape = {model.output_shape}')

    #------------------LOAD_VAL_DATA------------------

    # data_path = '/home/ml/datasets/mars/price_tags/{}.npy'
    # data_path = '/home/ml/datasets/schwarzkopf-retail/price_tags/{}.npy'

    # val_images = np.load(data_path.format('val_x'))
    # val_annotations = np.load(data_path.format('val_y'))

    # val_images = np.load(data_path.format('val_x_mars_schwarzkopf'))
    # val_annotations = np.load(data_path.format('val_y_mars_schwarzkopf'))

    # dataset_path = '/home/ml/datasets/schwarzkopf-retail/price_tags'
    # val_images = np.load(f'{dataset_path}/val_x_mars_schwarzkopf_14282.npy')
    # val_annotations = np.load(f'{dataset_path}/val_y_mars_schwarzkopf_14282.npy')

    # dataset_path = '/home/ml/datasets/schwarzkopf-retail/price_tags'
    # val_images = np.load(f'{dataset_path}/val_x_mars_schwarzkopf_16577.npy')
    # val_annotations = np.load(f'{dataset_path}/val_y_mars_schwarzkopf_16577.npy')
    # val_filenames = np.load(f'{dataset_path}/val_filenames_mars_schwarzkopf_16577.npy')

    # filter
    # indexes = np.flatnonzero(['schwarzkopf-retail/price_tags' in i for i in val_filenames])
    # indexes = np.flatnonzero(['mars/price_tags' in i for i in val_filenames])
    # val_images = val_images[indexes]
    # val_annotations = val_annotations[indexes]
    # val_filenames = val_filenames[indexes]


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

    # val_indexes = filter_percent_data(val_annotations)

    # val_images = val_images[val_indexes]
    # val_annotations = val_annotations[val_indexes]

    # val_images = np.load(data_path.format('val_x_mars_schwarzkopf_small_digits'))
    # val_annotations = np.load(data_path.format('val_y_mars_schwarzkopf_small_digits'))
    # val_images = np.load(data_path.format('val_x_mars'))
    # val_annotations = np.load(data_path.format('val_y_mars'))
    # val_images = np.load(data_path.format('val_x_schwarzkopf'))
    # val_annotations = np.load(data_path.format('val_y_schwarzkopf'))


    img_height = input_shape[0]
    img_width = input_shape[1]

    convert_gray = ConvertColor(current='BGR', to='GRAY', keep_3ch=False, keep_1ch=True)
    box_filter_resize = BoxFilter(check_overlap=False,
                                  check_min_area=True,
                                  check_degenerate=True,
                                  min_area=3)
    resizer = Resize(img_height,
                     img_width,
                     interpolation_mode='custom',
                     box_filter=box_filter_resize,
                     keep_original_aspect_ratio=True,
                     align_mode=AlignMode.CENTER)

    resized_images = []
    resized_annotations = []
    for i in range(len(val_images)):
        img, ann = resizer(val_images[i], val_annotations[i])
        if count_channels == 1:
            img, ann = convert_gray(img, ann)
        resized_images.append(img)
        resized_annotations.append(ann)
    val_images = np.array(resized_images)
    val_annotations = np.array(resized_annotations)
    print(f'Data shepe: val_images {val_images.shape},val_annotations {val_annotations.shape}')
    y_pred = model.predict(val_images)
    print('Inference done')

    pairs = []

    for confidence_thresh in np.linspace(0.05,0.95,confidence_thresh_num):
        for iou_thresh in np.linspace(0.05,0.95,iou_thresh_num):
            pairs.append([confidence_thresh, iou_thresh])

    res = Parallel(n_jobs=os.cpu_count())(delayed(f)(input_shape=input_shape, y_pred=y_pred, y_true=val_annotations, confidence_thresh=i, iou_thresh=j) for i,j in pairs)

    np.save(res_filename, res)
    print('Validate done!')



def f(input_shape, y_pred, y_true, confidence_thresh, iou_thresh):
    # y_pred = np.array(decode_detections(y_pred,
    y_pred = np.array(decode_detections_fast(y_pred,
                                             normalize_coords=True,
                                             img_height=input_shape[0],
                                             img_width=input_shape[1],
                                             confidence_thresh=confidence_thresh,
                                             iou_threshold=iou_thresh))
    acc_micro, acc_macro, f1_micro, f1_macro, acc_list, f1_list = calc_acc_f1(y_true, y_pred, classs, gt_thresh=0.5)
    print(f'Done confidence_thresh: {confidence_thresh} iou_thresh: {iou_thresh}')
    return np.array([acc_micro, acc_macro, f1_micro, f1_macro, confidence_thresh, iou_thresh])



validate()
plot_detector_accuracy(res_filename, confidence_thresh_num, iou_thresh_num, show=False, savefig=False)
