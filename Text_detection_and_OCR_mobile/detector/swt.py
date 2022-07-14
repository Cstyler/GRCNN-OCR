import math

import cv2
import numpy as np

from price_detector.detector.utils import show_boxes_on_image


def swt_find_letters(img: np.ndarray, params: dict, show: bool):
    boxes = SWTLocalizerEipshtein.localize(img,
                                           params["threshold1"], params["threshold2"],
                                           params["swt_ratio_threshold"],
                                           params["max_anglediff"],
                                           params["connectivity"],
                                           show)
    if show:
        show_boxes_on_image(boxes, img)
    return boxes


# Implementation of disjoint-set
class Node:
    def __init__(self, value):
        self.value = value
        self.parent = self
        self.rank = 0


def MakeSet(x, disjoint_set):
    if x in disjoint_set:
        return disjoint_set[x]
    item = Node(x)
    disjoint_set[x] = item
    return item


def Find(item):
    if item.parent != item:
        item.parent = Find(item.parent)
    return item.parent


def Union(x, y):
    """
    :param x:
    :param y:
    :return: root node of new union tree
    """
    x_root = Find(x)
    y_root = Find(y)
    if x_root == y_root:
        return x_root

    if x_root.rank < y_root.rank:
        x_root.parent = y_root
        return y_root
    elif x_root.rank == y_root.rank:
        x_root.rank += 1
    y_root.parent = x_root
    return x_root


class SWTLocalizerEipshtein:
    @classmethod
    def localize(cls, img: np.ndarray, threshold1: int,
                 threshold2: int,
                 swt_ratio_threshold: int, max_anglediff: float,
                 connectivity: int,
                 show):
        """
        Apply Stroke-Width Transform to image.

        :param filepath: relative or absolute filepath to source image
        :return: numpy array representing result of transform
        """
        # canny, sobelx, sobely, grad_angle = cls._create_derivative(img, sigma)
        canny, sobelx, sobely = cls._calc_derivatives(img, threshold1,
                                                      threshold2)
        swt = cls._swt(canny, sobelx, sobely, max_anglediff)
        shapes = cls._connect_components(swt, swt_ratio_threshold, connectivity)
        return cls._find_letters(shapes)

    @classmethod
    def _calc_derivatives(cls, img, threshold1, threshold2, show=False):
        # apply automatic Canny edge detection using the computed median
        if isinstance(threshold1, float):
            v = np.median(img)
            threshold1 = int(max(0, (1. - threshold1) * v))
            threshold2 = int(min(255, (1. + threshold2) * v))
        edges = cv2.Canny(img, threshold1, threshold2, apertureSize=3)
        # def _create_derivative(cls, img, sigma):
        #     edges = auto_canny(img, sigma)
        # if show: show_img(edges)
        # Create gradient map using Sobel
        ddepth = cv2.CV_64F
        sobel_x = cv2.Sobel(img, ddepth, 1, 0)
        sobel_y = cv2.Sobel(img, ddepth, 0, 1)
        return edges, sobel_x, sobel_y

    @classmethod
    def _swt(self, edges, sobel_x, sobel_y, max_anglediff):
        # create empty image, initialized to infinity
        swt = np.empty(edges.shape)
        swt[:] = np.Infinity
        rays = []
        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        # step_grad_x = np.cos(-sobel_x)
        # step_grad_y = np.sin(-sobel_y)

        step_grad_x = -sobel_x
        step_grad_y = -sobel_y
        mag_g = np.sqrt(step_grad_x ** 2 + step_grad_y ** 2)
        step_grad_x = step_grad_x / mag_g
        step_grad_y = step_grad_y / mag_g

        h, w = edges.shape
        for x in range(w):
            for y in range(h):
                if edges[y, x] > 0:
                    grad_x_p = step_grad_x[y, x]
                    grad_y_p = step_grad_y[y, x]
                    if np.isnan(grad_x_p) or np.isnan(grad_y_p):
                        continue
                    self.swt_point(edges, grad_x_p, grad_y_p, step_grad_x, step_grad_y,
                                   rays, swt, x, y, h, w, max_anglediff)

        # Compute median SWT
        for ray in rays:
            sw_tupl = tuple(swt[y, x] for (x, y) in ray)
            median = np.median(sw_tupl)
            for (x, y), sw in zip(ray, sw_tupl):
                swt[y, x] = min(median, sw)

        return swt

    @classmethod
    def swt_point(cls, edges, grad_x_p, grad_y_p, step_grad_x, step_grad_y, rays, swt,
                  x_p, y_p, h, w, max_anglediff):
        ray = [(x_p, y_p)]
        prev_x, prev_y, i = x_p, y_p, 0
        while True:
            i += 1
            cur_x = math.floor(x_p + grad_x_p * i)
            cur_y = math.floor(y_p + grad_y_p * i)

            if cur_x != prev_x or cur_y != prev_y:
                # we have moved to the next pixel
                if 0 < cur_x < w and 0 < cur_y < h:
                    # found q point (on another edge)
                    if edges[cur_y, cur_x] > 0:
                        grad_coeff = -(grad_x_p * step_grad_x[cur_y, cur_x] + grad_y_p *
                                       step_grad_y[cur_y, cur_x])
                        if -1. < grad_coeff < 1. and \
                                not np.isnan(grad_coeff) and \
                                math.acos(grad_coeff) < max_anglediff:
                            p_q_distance = math.sqrt(
                                (cur_x - x_p) ** 2 + (cur_y - y_p) ** 2)
                            ray.append((cur_x, cur_y))
                            for (rp_x, rp_y) in ray:
                                swt[rp_y, rp_x] = min(p_q_distance,
                                                      swt[rp_y, rp_x])
                            rays.append(ray)
                        break
                    # this is positioned at end to ensure
                    # we don't add a point beyond image boundary
                    ray.append((cur_x, cur_y))
                else:
                    # reached image boundary
                    break
                prev_x = cur_x
                prev_y = cur_y

    @classmethod
    def _connect_components(cls, swt: np.ndarray, swt_ratio_threshold: float,
                            connectivity: int):
        # STEP: Compute distinct connected components
        # apply Connected Component algorithm, comparing SWT values.
        # components with a SWT ratio less extreme than 1:3 are assumed to be
        # connected. Apply twice, once for each ray direction/orientation, to
        # allow for dark-on-light and light-on-dark texts
        trees = {}
        disjoint_set = {}
        label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
        next_label = 1
        # First Pass, raster scan-style
        height, width = swt.shape
        for i in range(height):
            for j in range(width):
                sw_point = swt[i, j]
                if 0 < sw_point < np.Infinity:
                    if connectivity == 0:
                        neighbors = [(i, j - 1),  # west
                                     (i - 1, j - 1),  # northwest
                                     (i - 1, j),  # north
                                     (i - 1, j + 1)]  # northeast
                    elif connectivity == 1:
                        neighbors = [(i, j - 1),
                                     (i - 1, j),
                                     (i + 1, j),
                                     (i, j + 1)]
                    else:
                        raise ValueError(f"Connectivity:{connectivity} is not correct")
                    connected_neighbors = None
                    neighbor_vals = []

                    for neighbor in neighbors:
                        try:
                            sw_n = swt[neighbor]
                            label_n = label_map[neighbor]
                        except IndexError:
                            continue
                        if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and \
                                sw_point / sw_n < swt_ratio_threshold:
                            neighbor_vals.append(label_n)
                            if connected_neighbors is not None:
                                connected_neighbors = Union(connected_neighbors,
                                                            MakeSet(label_n,
                                                                    disjoint_set))
                            else:
                                connected_neighbors = MakeSet(label_n, disjoint_set)

                    if connected_neighbors is None:
                        # We don't see any connections to North/West
                        trees[next_label] = MakeSet(next_label, disjoint_set)
                        label_map[i, j] = next_label
                        next_label += 1
                    else:
                        # We have at least one connection to North/West
                        label_map[i, j] = min(neighbor_vals)
                        # For each neighbor, make note that
                        # their respective
                        # connected_neighbors are connected
                        # for label in connected_neighbors.
                        trees[connected_neighbors.value] = Union(
                            trees[connected_neighbors.value], connected_neighbors)

        # Second pass. re-base all labeling with
        # representative label for each connected tree
        layers = {}
        for j in range(width):
            for i in range(height):
                cur_label = label_map[i, j]
                if cur_label > 0:
                    item = disjoint_set[cur_label]
                    common_label = Find(item).value
                    label_map[i, j] = common_label
                    if common_label in layers:
                        layer = layers[common_label]
                    else:
                        layers[common_label] = np.zeros(shape=swt.shape, dtype=np.uint16)
                        layer = layers[common_label]
                    layer[i, j] = 1
        return layers

    @classmethod
    def _find_letters(cls, shapes: dict, min_size=5):
        # STEP: Discard shapes that are probably not letters
        boxes = []
        for layer in shapes.values():
            nz_y, nz_x = np.nonzero(layer)
            xmax, xmin, ymax, ymin = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
            width = xmax - xmin
            height = ymax - ymin
            if width < min_size or height < min_size:
                continue
            boxes.append(np.array((xmin, ymin, xmax, ymax)))
        return np.array(boxes)
