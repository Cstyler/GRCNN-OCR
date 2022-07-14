from tensorflow_core.python.keras.saving.save import load_model

from price_detector.boxes_pattern_match import PriceMatcher
from price_detector.detector.detect_digits import detect_digits, get_algo_params
from price_detector.detector.utils import show_boxes_on_image
from price_detector.recognizer import MODELS_DIR, digits_to_number


class PriceRecognizer2:
    def __init__(self, version="v6"):
        model_name = 'digits_epoch-74_loss-0.1696_acc-0.9559.h5'
        cluster_centers_file = f"cluster_centers"
        features_means_file = "features-means"
        features_scales_file = "features-scales"
        annotated_cluster_members_file = "cluster_member_annotations"
        self.max_side = 120
        self.area_filter_params, self.aspect_ratio_filter_params, \
        self.divide_algo_param_dict, self.mser_params, \
        self.threshold_dict, self.box_expand_size, self.swt_params = get_algo_params(
            version, True)
        self.digits_model = load_model(MODELS_DIR / model_name)
        self.rub_kop_finder = PriceMatcher(cluster_centers_file, features_means_file,
                                           features_scales_file,
                                           annotated_cluster_members_file)

    def recognize_float(self, img):
        pred_boxes = self.detect(img)
        if len(pred_boxes):
            h, w = img.shape[:2]
            try:
                rub, kop = self.rub_kop_finder.match(pred_boxes, w, h)
            except ValueError:
                return 0.
            return digits_to_number(rub) + digits_to_number(kop) / 100
        else:
            return 0.

    def recognize(self, img, show=False):
        pred_boxes = self.detect(img)
        if show: show_boxes_on_image(pred_boxes, img)
        if len(pred_boxes):
            h, w = img.shape[:2]
            rub, kop = self.rub_kop_finder.match(pred_boxes, w, h)
            if kop is not None:
                if len(kop) < 2:
                    kop += [0]
                kop_float = digits_to_number(kop) / 100
            else:
                kop_float = 0.
            rub_float = digits_to_number(rub)
            price_float = round(rub_float + kop_float, 2)
            return price_float, pred_boxes
        else:
            return 0., []

    def detect(self, img):
        return detect_digits(img, self.digits_model, self.mser_params,
                             self.area_filter_params, self.aspect_ratio_filter_params,
                             self.threshold_dict, self.box_expand_size, None,
                             self.max_side, False,
                             divide_algo_param_dict=self.divide_algo_param_dict)
