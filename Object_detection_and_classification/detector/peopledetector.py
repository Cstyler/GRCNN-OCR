import os
import cv2
from models.background import BackgroundClassifier
from myutils import draw_rectange


class PeopleDetector:
    def __init__(self, backgroundimage, roi, image_per_background=5,
                 occupancy_thresh=0.5, blur=True, debug=True, weights=None):
        self.__image_per_background = image_per_background
        self.__occupancy_thresh = occupancy_thresh
        self.__roi = roi
        self.__blur = blur
        self.__debug = True
        self.__classifier = BackgroundClassifier(weights=weights)
        self.__man_is_here = None
        self.__fgmask = None
        self.__frame = None

        self.__mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.__mog.apply(backgroundimage, learningRate=1)

    @property
    def mask(self):
        return self.__fgmask

    @property
    def is_debug(self):
        return self.__debug

    @property
    def roi_img(self):
        roi = self.__roi
        return self.__frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].copy()

    def process_frame(self, frame):
        self.__frame = frame.copy()
        img = self.roi_img

        if self.__image_per_background:
            if self.__classifier.predict(img) == 0:
                if self.__blur:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                self.__mog.apply(img)
                self.__image_per_background -= 1

            return False

        if self.__blur:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        fgmask = self.__mog.apply(img, learningRate=0)
        occupancy = fgmask.sum() / (fgmask.size * 255)

        self.__fgmask = fgmask
        self.__man_is_here = occupancy > self.__occupancy_thresh
        return self.__man_is_here

    def debug(self, frame, draw_roi=True, label=True):
        if self.__man_is_here and label:
            cv2.putText(frame, "MAN IS HERE", (frame.shape[1]-300, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1,
                        cv2.LINE_AA)

        if draw_roi:
            roi = self.__roi
            frame = draw_rectange(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]))

        return frame
