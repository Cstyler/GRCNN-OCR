import numpy as np

np.random.seed(42)
import cv2
from .request_utils import load_image, GoogleVisionRequester
from .exceptions import BadRequestError
from .types import ResponseType, BlocksType
from itertools import chain
from typing import Iterable, Tuple, List, Union, Optional
import re


def _remove_nonnumeric_from_str(s: str) -> str:
    pattern = r"[^\d]"
    return re.sub(pattern, "", s)


ShapeType = Tuple[int, int]


class Recognizer:
    name_shape = (650, 650)
    price_shape = (20, 40)
    code_shape = (250, 250)

    m_name, n_name = 14, 1
    m_price, n_price = 4, 4
    m_code, n_code = 14, 1

    num_of_white_rects = 1

    price_pad_value = 180
    code_pad_value = 180
    name_pad_value = 180

    name_font_scale = 3
    price_font_scale = 0.75
    code_font_scale = 1
    base_symbol = 'X'
    splitter_symbols = {'X', 'Х', 'x', 'х'}

    unknown_product_code_type = "unknown"
    ean8_product_code_type = 'EAN_8'
    ean13_product_code_type = 'EAN_13'
    ean8_len = 8
    ean13_len = 13

    price_segment_names = frozenset(
            ('price_rub', 'price_kop', 'price_rub_discount', 'price_kop_discount',
             'price_rub_card', 'price_kop_card'))

    def __init__(self):
        self.google_requester = GoogleVisionRequester()

    @staticmethod
    def get_symbols_from_blocks(blocks: BlocksType, x1: int,
                                y1: int, x2: int, y2: int,
                                epsilon0: int, epsilon1: int) -> str:
        res_text = []
        for block in blocks:
            words = block['paragraphs'][0]['words']
            for word in words:
                symbols = word['symbols']
                for symbol in symbols:
                    v1, _, v2, _ = symbol['boundingBox']['vertices']
                    try:
                        x1_, y1_ = v1['x'], v1['y']
                        x2_, y2_ = v2['x'], v2['y']
                    except KeyError:
                        continue
                    if abs(x1 - x1_) < epsilon1 and abs(x2 - x2_) < epsilon1 and abs(
                            y1 - y1_) < epsilon0 and abs(y2 - y2_) < epsilon0:
                        text = symbol['text']
                        try:
                            break_type = symbol['property']['detectedBreak']['type']
                            if break_type == 'SPACE' or break_type == "EOL_SURE_SPACE":
                                text += " "
                        except KeyError:
                            pass
                        res_text.append(text)
        res_text = "".join(res_text)
        return res_text

    def draw_text(self, img: np.ndarray, shape: ShapeType, font_scale: float) -> None:
        color = 0
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        line_type = cv2.LINE_4
        bottom_left_origin = False
        w, h = shape
        (text_w, text_h), base_line = cv2.getTextSize(
                self.base_symbol, font_face, font_scale, thickness)
        org = (h - text_w) // 2, (w + text_w) // 2
        cv2.putText(img, self.base_symbol, org, font_face, font_scale, color,
                    thickness, line_type, bottom_left_origin)

    def check_symbols_recognized(self, response: ResponseType,
                                 num_of_crops: int) -> Tuple[bool, str]:
        text = response['textAnnotations'][0]['description']
        num_splitters = sum(x in self.splitter_symbols for x in text)
        flag1 = num_splitters == num_of_crops
        newline_count = text.count("\n")
        expected_num_of_newline = int(num_of_crops // self.m_price + (num_of_crops % self.m_price != 0))
        # expected: for 1 <= num_of_crops < 5 is 1, for 5 <= num_of_crops < 9 is 2 etc...
        flag2 = expected_num_of_newline == newline_count
        symbols_recognized = flag1 and flag2
        return symbols_recognized, text

    def iterate_text_from_response(self, response: ResponseType,
                                   num_of_crops: int, shape: ShapeType,
                                   m: int, n: int) -> Iterable[str]:
        if n != 1:
            try:
                symbols_recognized, text = \
                    self.check_symbols_recognized(response, num_of_crops)
                if symbols_recognized:
                    for x in self.text_parse_algo(text):
                        yield x
                    return
            except (KeyError, IndexError):
                for _ in range(num_of_crops):
                    yield None
                return
            for x in self.coordinate_algo1(response, num_of_crops, shape, m, n):
                yield x
        else:
            for x in self.coordinate_algo2(response, num_of_crops, shape, m, n):
                yield x

    def coordinate_algo2(self, response: ResponseType,
                        num_of_crops: int, shape: ShapeType,
                        m: int, n: int) -> Iterable[str]:
        try:
            blocks = response['fullTextAnnotation']['pages'][0]['blocks']
        except (KeyError, IndexError):
            for _ in range(num_of_crops):
                yield None
            return
        k = 0
        break_flag = False
        h, w = shape
        epsilon0 = h
        epsilon1 = w
        for j in range(m):
            if break_flag:
                break
            for i in range(n):
                x1, y1 = 0, j * h
                x2, y2 = w, y1 + h
                res_text = self.get_symbols_from_blocks(blocks,
                                                        x1, y1, x2, y2, epsilon0, epsilon1)
                k += 1
                if k > num_of_crops:
                    break_flag = True
                    break
                yield res_text

    def coordinate_algo1(self, response: ResponseType,
                        num_of_crops: int, shape: ShapeType,
                        m: int, n: int) -> Iterable[str]:
        try:
            blocks = response['fullTextAnnotation']['pages'][0]['blocks']
        except (KeyError, IndexError):
            for _ in range(num_of_crops):
                yield None
            return
        k = 0
        break_flag = False
        h, w = shape
        epsilon0 = self.num_of_white_rects * h
        epsilon1 = self.num_of_white_rects * w
        for j in range(m):
            if break_flag:
                break
            for i in range(n):
                x1, y1 = self.num_of_white_rects * w + i * \
                         (w * (self.num_of_white_rects + 1)), self.num_of_white_rects * \
                         h + j * (h * (self.num_of_white_rects + 1))
                x2, y2 = x1 + w, y1 + h
                res_text = self.get_symbols_from_blocks(blocks,
                                                        x1, y1, x2, y2, epsilon0, epsilon1)
                k += 1
                if k > num_of_crops:
                    break_flag = True
                    break
                yield res_text

    def text_parse_algo(self, text: str) -> List[str]:
        remove_symbols = {' ', '\n'}
        text = "".join('' if x in remove_symbols else
                       (self.base_symbol if x in self.splitter_symbols else x) for x in text)
        tag_data_list = text.split(self.base_symbol)[:-1]
        return tag_data_list

    def join_crops1(self, crops: List[np.ndarray], shape: ShapeType, font_scale: float,
                    m: int, n: int, pad_value: int) -> np.array:
        def get_crop(crop_ind):
            crop = crops[crop_ind]

            if crop is None:
                return white_place_holder
            return crop

        h, w = shape
        white_rect = pad_value * np.ones((h, int(self.num_of_white_rects * w)))
        white_rect_with_text = white_rect.copy()
        self.draw_text(white_rect_with_text, white_rect.shape, font_scale)
        white_row = pad_value * np.ones((int(self.num_of_white_rects * h),
                                         int(self.num_of_white_rects * w
                                             + n * (1 + self.num_of_white_rects) * w)))
        k = 0
        rows = [white_row]
        white_place_holder = pad_value * np.ones(shape)
        for i in range(m):
            split_rect = white_rect if crops[k] is None else white_rect_with_text
            img = get_crop(k)
            k += 1
            row = img
            row = np.concatenate((row, split_rect), axis=1)
            row = np.concatenate((white_rect, row), axis=1)
            for j in range(1, n):
                split_rect = white_rect if crops[k] is None else white_rect_with_text
                img = get_crop(k)
                k += 1
                row = np.concatenate((row, img), axis=1)
                row = np.concatenate((row, split_rect), axis=1)

            rows.append(row)
            rows.append(white_row)
        img = np.concatenate(rows, axis=0)
        return img

    def join_crops_vertical(self, crops: List[np.ndarray], shape: int, pad_value: int) -> np.array:
        padding_frac = .1
        padding = int(padding_frac * shape)
        img_list = []
        resize_to = shape - 2 * padding
        empty_array = pad_value * np.ones((shape, shape))
        more_than_one_flag = len(crops) > 1
        for crop in crops:
            img = empty_array.copy() if more_than_one_flag else empty_array
            if crop is not None:
                h, w = crop.shape[:2]
                if w > h:
                    h = int(resize_to * h / w)
                    w = resize_to
                else:
                    w = int(resize_to * w / h)
                    h = resize_to
                crop = cv2.resize(crop, (w, h))
                x = int((shape - w) / 2)
                y = int((shape - h) / 2)
                img[y:y + h, x:x + w] = crop
            img_list.append(img)
        img = np.concatenate(img_list, axis=0)
        return img

    def create_joined_crops_list(self, m: int, n: int,
                                 shape: ShapeType, font_scale: float,
                                 crops: List[np.array],
                                 pad_value: int) -> Tuple[List[np.ndarray], List[int]]:
        def add_to_list(_chunk_size: int, chunk: list) -> None:
            num_of_crops.append(_chunk_size)
            if n != 1:
                img = self.join_crops1(
                        chunk, shape, font_scale, m, n, pad_value)
            else:
                img = self.join_crops_vertical(
                        chunk, shape[0], pad_value)
            images.append(img)

        images, num_of_crops = [], []
        chunk_size = m * n
        crops_len = int(len(crops) / chunk_size) * chunk_size
        if len(crops) >= chunk_size:
            for i in range(0, crops_len, chunk_size):
                crops_chunk = crops[i:i + chunk_size]
                add_to_list(chunk_size, crops_chunk)

        last_chunk_size = len(crops[crops_len:])
        if last_chunk_size:
            last_chunk = crops[crops_len:] + \
                         [None for _ in range(chunk_size - last_chunk_size)]
            add_to_list(last_chunk_size, last_chunk)
        return images, num_of_crops

    def iterate_tag_data_from_responses(self, responses: List[ResponseType], num_of_crops: List[int],
                                        shape: ShapeType,
                                        m: int, n: int) -> Iterable[str]:
        iter = chain(
                *(self.iterate_text_from_response(response, num, shape, m, n) for
                  response, num in zip(responses, num_of_crops)))
        return iter

    @staticmethod
    def load_crop_of_images(sources: Iterable[str],
                            src_type: str, shape: ShapeType, resize_flag: bool) -> List[np.ndarray]:
        images = []
        for src in sources:
            img = load_image(src, src_type)
            if resize_flag:
                img = cv2.resize(img, shape)
            images.append(img)
        return images

    def concrete_data_process(self, sources: Iterable[str],
                              src_type: str, m: int, n: int, shape: ShapeType,
                              font_scale: float,
                              pad_value: int) -> Iterable[str]:
        shape_ = (shape[1], shape[0])
        resize_flag = n != 1
        crops = self.load_crop_of_images(sources, src_type, shape_, resize_flag)
        joined_crop_images, num_of_crops = self.create_joined_crops_list(
                m, n, shape, font_scale, crops, pad_value=pad_value)
        response = self.google_requester.get_responses(joined_crop_images)
        if not response.ok:
            raise BadRequestError('Google returned "not ok" response: %s' % response.json())
        responses = response.json()['responses']
        return self.iterate_tag_data_from_responses(responses, num_of_crops, shape, m, n)

    def recognize_concrete_data(self, sources: Iterable[str],
                                src_type: str, data_type: str) -> Iterable[Union[str, Tuple[str, str]]]:
        if data_type == 'text':
            iter = self.concrete_data_process(sources, src_type,
                                              self.m_name, self.n_name,
                                              self.name_shape,
                                              self.name_font_scale, self.name_pad_value)
            for x in iter:
                yield x
        elif data_type in self.price_segment_names:
            iter = self.concrete_data_process(sources, src_type,
                                              self.m_price, self.n_price,
                                              self.price_shape,
                                              self.price_font_scale, self.price_pad_value)
            for x in iter:
                yield x
        else:
            iter = self.concrete_data_process(sources, src_type,
                                              self.m_code, self.n_code,
                                              self.code_shape,
                                              self.code_font_scale, self.code_pad_value)
            for code in iter:
                yield self.recognize_product_code_type(code)

    def recognize_product_code_type(self, code: str) -> Tuple[str, str]:
        if not code:
            return code, self.unknown_product_code_type
        code = _remove_nonnumeric_from_str(code)
        if not code:
            return code, self.unknown_product_code_type
        checksum = self.calc_checksum_of_code(code)
        original_checksum = int(code[-1])
        if checksum is not None and original_checksum == checksum:
            if len(code) == 8:
                return code, self.ean8_product_code_type
            else:
                return code, self.ean13_product_code_type
        return code, self.unknown_product_code_type

    def calc_checksum_of_code(self, code: str) -> Optional[int]:
        if len(code) not in {self.ean8_len, self.ean13_len}:
            return
        code = code.zfill(self.ean13_len)
        sum1 = sum(int(code[i]) for i in range(1, 12, 2))  # sum even indices
        sum1 *= 3
        sum2 = sum(int(code[i]) for i in range(0, 11, 2))  # sum odd indices
        total = sum1 + sum2
        checksum = total % 10
        if checksum:
            checksum = 10 - checksum

        return checksum
