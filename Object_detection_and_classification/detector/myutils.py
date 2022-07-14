import json
import cv2


def read_config():
    with open('config.json') as json_file:
        config = json.load(json_file)
    return config


def draw_rectange(image, startpoint, endpoint):
    image = cv2.rectangle(image, startpoint, endpoint, (255, 255, 0), 5)
    return image


def put_text(image, text, point):
    image = cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image
