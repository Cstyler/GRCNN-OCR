import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import cv2
from PIL import Image


class BackgroundClassifier:

    def __init__(self, weights=None, height=224, width=224, channels=3, thresh=0.5):
        self.model = load_model(weights)
        self.height = height
        self.width = width
        self.channels = channels
        self.thresh = thresh

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height))
        img = preprocess_input(img)
        img = img.reshape(1, self.width, self.height, self.channels)
        return int(self.model.predict(img)[0] > self.thresh)
