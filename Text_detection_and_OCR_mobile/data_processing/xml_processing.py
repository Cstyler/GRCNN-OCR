from pathlib import Path
from typing import Tuple
from xml.etree import ElementTree as ET

CLASSES_DICT = {'1': 2,
                '2': 3,
                '3': 4,
                '4': 5,
                '5': 6,
                '6': 7,
                '7': 8,
                '8': 9,
                '9': 10,
                '0': 1,
                '%': 11}


def parse_xml(file_: Path):
    try:
        root = ET.parse(file_).getroot()
    except ET.ParseError:
        return None, None, None
    image_shape = parse_xml_shape(root)
    filename = get_xml_field(root, "filename")
    boxes = parse_xml_boxes(root, image_shape, filename)
    tag_id = Path(filename).stem
    return tag_id, boxes, image_shape


def parse_xml_boxes(root: ET, image_shape: Tuple[int, int], filename: str):
    boxes = []
    img_h, img_w = image_shape

    for box_node in root.iter("object"):
        class_name = get_xml_field(box_node, "name")

        if class_name not in CLASSES_DICT:
            return
        class_ = CLASSES_DICT[class_name]

        box_node = box_node.find("bndbox")

        ymin = int(get_xml_field(box_node, "ymin"))
        xmin = int(get_xml_field(box_node, "xmin"))
        ymax = int(get_xml_field(box_node, "ymax"))
        xmax = int(get_xml_field(box_node, "xmax"))

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, img_w)
        ymax = min(ymax, img_h)

        if xmin >= xmax or ymin >= ymax:
            print(f'Founded degenerate box:'
                  f' {(xmin, ymin, xmax, ymax)} '
                  f'\'class_name\' = \'{class_name}\' \'filename\' = \'{filename}\'!')
            return
        if xmin >= img_w or ymin >= img_h:
            print(f'Founded box outside the image:'
                  f' {(xmin, ymin, xmax, ymax)}'
                  f' \'class_name\' = \'{class_name}\' \'filename\' = \'{filename}\'!')
            return

        box = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        box["class"] = class_
        boxes.append(box)
    if boxes:
        return boxes


def parse_xml_shape(root: ET):
    size = root.find("size")
    return int(get_xml_field(size, "height")), int(get_xml_field(size, "width"))


def get_xml_field(obj: ET, field: str):
    return obj.find(field).text
