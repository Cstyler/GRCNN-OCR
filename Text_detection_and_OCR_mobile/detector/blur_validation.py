import cv2
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

from tools.image_utils import resize_image
from tools.inference import __convert_grayscale_image_for_input_layer, load_keras_model


def prepare_input_data(images_or_filenames,
                       input_h,
                       input_w,
                       input_c,
                       interpolation,
                       keep_original_aspect_ratio,
                       align_center,
                       background):
    '''
    Check images_or_filenames type, read and resize data.
    :param images_or_filenames:
    :param input_h:
    :param input_w:
    :param input_c:
    :param interpolation:
    :param keep_original_aspect_ratio:
    :param align_center
    :param background
    :return:
    '''

    # check if images is empty
    if (isinstance(images_or_filenames, np.ndarray) and images_or_filenames.shape[0] == 0) \
            or (isinstance(images_or_filenames, list) and len(images_or_filenames) == 0):
        return np.array([]), np.array([])

    # if images is ndarray or list filenames
    if isinstance(images_or_filenames[0], str):
        images_or_filenames = Parallel(n_jobs=-1)(
            delayed(cv2.imread)(img) for img in images_or_filenames)

    # get default images_shapes (add 1 channel for grayscale)
    images_shapes = np.array(
        [img.shape if img.ndim == 3 else list(img.shape) + [1] for img in
         images_or_filenames])
    if any(images_shapes[:, 2] != input_c):
        raise Exception(f'Image depth and network = {input_c} do not match!')

    # resize
    images_or_filenames = Parallel(n_jobs=-1)(
        delayed(resize_image)(img, input_h, input_w, interpolation,
                              keep_original_aspect_ratio, align_center, background) for
        img in images_or_filenames)
    images_or_filenames = Parallel(n_jobs=-1)(
        delayed(__convert_grayscale_image_for_input_layer)(img) for img in
        images_or_filenames)
    images_or_filenames = np.array(images_or_filenames)

    return images_or_filenames, images_shapes


class BlurValidator:
    def __init__(self, model=None, custom_objects=None):
        if model is None:
            model = '/home/ml/models/mlflow-artifact/35/' \
                    'b9aa24953a604feaa16c11f597815c2f/artifacts' \
                    '/blur_classifier_3_epoch-254_loss-0.2789_' \
                    'val_loss-0.2578_val_acc-0.9095.tflite'
        if '.tflite' in model:
            # Load TFLite model and allocate tensors.
            self.interpreter = tf.lite.Interpreter(model_path=model)
            self.interpreter.allocate_tensors()
            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.tflite = True
        else:
            self.tflite = False
            self.model = load_keras_model(model, custom_objects)

    def infer_crossentropy_classification(self, images_or_filenames,
                                          interpolation='custom',
                                          keep_original_aspect_ratio=False,
                                          align_center=False,
                                          background=(0, 0, 0)):
        '''
        :param images_or_filenames: ndarray images (uint8) or ndarray filenames (str) or list filenames (str)
        :param model: instance of loaded model or model filepath
        :param custom_objects: custom_objects for load model (need if model is filepath)
        :param interpolation: interpolation mode for resize image
        :param keep_original_aspect_ratio: keep aspect ratio when resizing
        :param align_center:
        :param background:
        :return: return ndarray of classes ids
        '''

        # prepare input data before infer

        input_h, input_w, input_c = self.model.input_shape[1:] \
            if not self.tflite and isinstance(self.model, tf.keras.Model) else \
            self.input_details[0]['shape'][1:]
        images_or_filenames, images_shapes = prepare_input_data(images_or_filenames,
                                                                input_h,
                                                                input_w,
                                                                input_c,
                                                                interpolation,
                                                                keep_original_aspect_ratio,
                                                                align_center,
                                                                background)

        if images_or_filenames.shape[0] == 0:
            return images_or_filenames

        if self.tflite:
            pred = []
            for element in images_or_filenames:
                self.interpreter.set_tensor(self.input_details[0]['index'], np.float32([
                    element]))
                self.interpreter.invoke()
                pred.append(
                    self.interpreter.get_tensor(self.output_details[0]['index'])[0])
            pred = np.array(pred)
        else:
            pred = self.model.predict(images_or_filenames)

        classes = pred.argmax(axis=1)

        return classes
