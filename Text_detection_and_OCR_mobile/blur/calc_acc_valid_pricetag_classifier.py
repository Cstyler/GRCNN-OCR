from tools.utils import enableGPU, allow_growth, disableGPU
# enableGPU(3)
# allow_growth()
disableGPU()

from sku_classifier.validation import calc_f1
from tools.inference import calc_metrics
import numpy as np
import cv2
import os
import glob
from tools.inference import infer_crossentropy_classification
from SkynetCV import SkynetCV

# trained on validity_dataset_992
# model_path = '/home/oleynik/saved_models/blur_classifier_3_epoch-254_loss-0.2789_val_loss-0.2578_val_acc-0.9095.h5'
# validity_dataset_992
# calc_f1 [{'class_id': 0.0, 'f1': 0.8944723618090453}, {'class_id': 1.0, 'f1': 0.9292929292929292}]
# metrics_sum, metrics_per_classes [{'precision_micro': 0.9153225806451613, 'recall_micro': 0.9153225806451613, 'f1_micro': 0.9153225806451613, 'business_acc_micro': 0.8438661710037175}, {'precision_macro': 0.9179539286678928, 'recall_macro': 0.907676673901316, 'f1_macro': 0.9127863737552925, 'business_acc_macro': 0.838507718696398}] [{'class_id': 0.0, 'class_count': 413, 'precision': 0.9295039164490861, 'recall': 0.8619854721549637, 'f1': 0.8944723618090453, 'business_acc': 0.8090909090909091}, {'class_id': 1.0, 'class_count': 579, 'precision': 0.9064039408866995, 'recall': 0.9533678756476683, 'f1': 0.9292929292929292, 'business_acc': 0.8679245283018868}]
# validity_dataset_5235
# calc_f1 [{'class_id': 0.0, 'f1': 0.8062283737024221}, {'class_id': 1.0, 'f1': 0.8467328087581253}]
# metrics_sum, metrics_per_classes [{'precision_micro': 0.8288443170964661, 'recall_micro': 0.8288443170964661, 'f1_micro': 0.8288443170964661, 'business_acc_micro': 0.707714891534823}, {'precision_macro': 0.8551853214036511, 'recall_macro': 0.8319864957972418, 'f1_macro': 0.8434264152091897, 'business_acc_macro': 0.7047829096427758}] [{'class_id': 0.0, 'class_count': 2678, 'precision': 0.9578622816032888, 'recall': 0.6960418222554144, 'f1': 0.8062283737024221, 'business_acc': 0.6753623188405797}, {'class_id': 1.0, 'class_count': 2557, 'precision': 0.7525083612040134, 'recall': 0.9679311693390692, 'f1': 0.8467328087581253, 'business_acc': 0.7342035004449718}]

# trained on validity_dataset_5235
model_path = '/home/oleynik/saved_models/blur_classifier_5_epoch-361_loss-0.2574_val_loss-0.2324_val_acc-0.9189.h5'
# validity_dataset_992
# calc_f1 [{'class_id': 0.0, 'f1': 0.8023668639053255}, {'class_id': 1.0, 'f1': 0.8533801580333628}]
# metrics_sum, metrics_per_classes [{'precision_micro': 0.8316532258064516, 'recall_micro': 0.8316532258064516, 'f1_micro': 0.8316532258064515, 'business_acc_micro': 0.7118205349439172}, {'precision_macro': 0.8262896825396826, 'recall_macro': 0.8301007414470135, 'f1_macro': 0.8281908277099735, 'business_acc_macro': 0.7071088742138746}] [{'class_id': 0.0, 'class_count': 413, 'precision': 0.7847222222222222, 'recall': 0.8208232445520581, 'f1': 0.8023668639053255, 'business_acc': 0.6699604743083004}, {'class_id': 1.0, 'class_count': 579, 'precision': 0.8678571428571429, 'recall': 0.8393782383419689, 'f1': 0.8533801580333628, 'business_acc': 0.7442572741194488}]
# validity_dataset_5235
# calc_f1 [{'class_id': 0.0, 'f1': 0.9139047619047618}, {'class_id': 1.0, 'f1': 0.9134099616858238}]
# metrics_sum, metrics_per_classes [{'precision_micro': 0.913658070678128, 'recall_micro': 0.913658070678128, 'f1_micro': 0.913658070678128, 'business_acc_micro': 0.8410409706347811}, {'precision_macro': 0.9139840560319428, 'recall_macro': 0.914080181715001, 'f1_macro': 0.9140321163461682, 'business_acc_macro': 0.8410398647642502}] [{'class_id': 0.0, 'class_count': 2678, 'precision': 0.9327371695178849, 'recall': 0.8958177744585512, 'f1': 0.9139047619047618, 'business_acc': 0.8414591371448614}, {'class_id': 1.0, 'class_count': 2557, 'precision': 0.8952309425460008, 'recall': 0.932342588971451, 'f1': 0.9134099616858238, 'business_acc': 0.840620592383639}]



# root_dataset_dir = '/home/ml/datasets/price_tags/validity_dataset_992'
root_dataset_dir = '/home/ml/datasets/price_tags/validity_dataset_5235'
train_filenames_path = os.path.join(root_dataset_dir, 'train_filenames.npy')
train_annotations_path = os.path.join(root_dataset_dir, 'train_annotations.npy')
val_filenames_path = os.path.join(root_dataset_dir, 'val_filenames.npy')
val_annotations_path = os.path.join(root_dataset_dir, 'val_annotations.npy')

train_filenames = np.load(train_filenames_path)
train_annotations = np.load(train_annotations_path)
val_filenames = np.load(val_filenames_path)
val_annotations = np.load(val_annotations_path)

x = np.concatenate([train_filenames, val_filenames])
y = np.concatenate([train_annotations, val_annotations])
# x = np.concatenate([val_filenames])
# y = np.concatenate([val_annotations])

y_pred = infer_crossentropy_classification(x, model_path)


print('calc_f1', calc_f1(y, y_pred))

metrics_sum, metrics_per_classes = calc_metrics(y, y_pred)
print('metrics_sum, metrics_per_classes', metrics_sum, metrics_per_classes)