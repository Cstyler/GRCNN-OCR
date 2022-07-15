import base64
from typing import Any, BinaryIO, Dict, List, Optional

import cv2
import numpy as np
import requests

from pylibs import jpeg_utils
from .exceptions import BadRequestError, NotFoundError

image_load_session = requests.Session()


def load_image(src: str, src_type: str = "uri", color: bool = False) -> Optional[np.ndarray]:
    color_flag = int(color)  # 0 means grayscale in opencv
    if src_type == "uri":
        try:
            response = image_load_session.get(src)
        except UnicodeDecodeError:
            raise NotFoundError("Wrong URL: %s. Unicode decode error" % src)
        except requests.exceptions.ConnectionError:
            raise NotFoundError("Connection error. Might be wrong URL")
        except requests.exceptions.InvalidSchema:
            raise NotFoundError("Wrong URL: %s. Invalid schema of URL" % src)
        except requests.exceptions.MissingSchema:
            raise NotFoundError("Wrong URL: %s. Missing schema of URL" % src)
        except requests.exceptions.ReadTimeout:
            raise NotFoundError("Connection error. Timeout")
        except requests.exceptions.ChunkedEncodingError:
            raise NotFoundError("Chunked encoding error")
        except requests.exceptions.RequestException:
            raise NotFoundError("Unknown exception when loading image")

        if not response.ok:
            raise NotFoundError("Wrong URL: %s" % src)
        img = cv2.imdecode(np.array(bytearray(response.content), dtype=np.uint8), color_flag)
        assert_img_is_valid(img, src)
        return img
    elif src_type == "file":
        img = jpeg_utils.read_jpeg(src, color_space='rgb' if color_flag else 'gray')
        assert_img_is_valid(img, src)
        return img
    raise BadRequestError("Wrong src_type %s" % src_type)


def load_image_to_file(src: str, _file: BinaryIO) -> None:
    chunk_size = 32768
    try:
        with image_load_session.get(src, stream=True) as response:
            if not response.ok:
                raise NotFoundError("Wrong URL: %s" % src)
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    _file.write(chunk)
    except UnicodeDecodeError:
        raise NotFoundError("Wrong URL: %s. Unicode decode error" % src)
    except requests.exceptions.ConnectionError:
        raise NotFoundError("Connection error. Might be wrong URL")
    except requests.exceptions.InvalidSchema:
        raise NotFoundError("Wrong URL: %s. Invalid schema of URL" % src)
    except requests.exceptions.MissingSchema:
        raise NotFoundError("Wrong URL: %s. Missing schema of URL" % src)
    except requests.exceptions.ReadTimeout:
        raise NotFoundError("Connection error. Timeout")
    except requests.exceptions.ChunkedEncodingError:
        raise NotFoundError("Chunked encoding error")
    except requests.exceptions.RequestException:
        raise NotFoundError("Unknown exception when loading image")


def assert_img_is_valid(img: np.ndarray, src: str) -> None:
    if img is None:
        raise NotFoundError("Wrong image file: %s" % src)


class GoogleVisionRequester:
    api_key = ""
    url = "https://vision.googleapis.com/v1/images:annotate?key=%s" % api_key

    def __init__(self) -> None:
        session = requests.Session()
        self.session = session

    def get_responses(self, images: List[np.ndarray]) -> requests.Response:
        reqs = []
        for image in images:
            content = self.get_encoded_image(image)
            reqs.append(self.request_dict(content))
        request = dict(requests=reqs)
        try:
            response = self.session.post(self.url, json=request)
        except requests.exceptions.ConnectionError:
            raise NotFoundError("Connection error. Can't make post request to Google Vision API. Please try again")
        return response

    @staticmethod
    def get_encoded_image(image: np.ndarray) -> str:
        ret_val, buffer = cv2.imencode(".jpg", image)
        content = base64.b64encode(buffer).decode("UTF-8")
        return content

    @staticmethod
    def request_dict(content: str) -> Dict[str, Any]:
        return dict(image=dict(content=content),
                    features=[dict(type="TEXT_DETECTION")], imageContext=dict(languageHints=['en', 'ru']))
