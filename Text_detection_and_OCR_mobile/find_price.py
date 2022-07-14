import operator

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from price_detector.data_processing import min_max_boxes_to_center_boxes, \
    normalize_features
from price_detector.detector.box_utils import box_area


# расчет расстояния между двумя точками
def dist(x1, y1, x2, y2) -> float:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# расчет угла в градусах между прямыми по трем точкам
#        1 ^
#         /
#        /
#       /
#      /
#   2 / angle     3
#    ------------->
def calc_angle_degrees(x1, y1, x2, y2, x3, y3) -> float:
    x1, y1 = x1 - x2, y1 - y2
    x3, y3 = x3 - x2, y3 - y2
    d1 = dist(0, 0, x1, y1)
    d3 = dist(0, 0, x3, y3)
    return np.arccos((x1 * x3 + y1 * y3) / (d1 * d3)) * 180. / np.pi


def get_line_segment(xmin, ymin, xmax, ymax, left):
    return ((xmin, ymin), (xmin, ymax)) if left else ((xmax, ymin), (xmax, ymax))


def box_line_segment_distance(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2,
                              direction):
    left = direction == 'left'
    p_a, p_b = get_line_segment(xmin1, ymin1, xmax1, ymax1, left)
    p_c, p_d = get_line_segment(xmin2, ymin2, xmax2, ymax2, not left)
    distance = line_segment_distance(p_a, p_b, p_c, p_d)
    return distance


def get_line_slope_shift(x1, y1, x2, y2):
    if x2 == x1:
        return None, None
    slope = (y2 - y1) / (x2 - x1)
    shift = -x1 * slope + y1
    return slope, shift


def point_is_between_two_points_on_line_segment(xa, ya, xb, yb, xc, yc,
                                                epsilon=.001):
    a1 = xb - xa
    a2 = xc - xa
    b1 = yb - ya
    b2 = yc - ya
    crossproduct = b2 * a1 - a2 * b1
    sq_len = a1 ** 2 + b1 ** 2
    dotproduct = a2 * a1 + b2 * b1
    return abs(
        crossproduct) <= epsilon and dotproduct >= 0 and dotproduct <= sq_len


def line_segment_distance(point_a, point_b, point_c, point_d):
    # Координаты концов первого отрезка: A(xa, ya), B(xb, yb).
    # Координаты концов второго отрезка: C(xc, yc), D(xd, yd).

    def height_end_x_point(x1, y1, x2, y2, x3, y3, k, d):
        den = x3 * x2 - x3 * x1 + y2 * y3 - y1 * y3 + y1 * d - y2 * d
        num = k * (y2 - y1) + x2 - x1
        return den / num

    def add_height_len_to_list(x1, y1, x2, y2, x3, y3, k, d):
        xz = height_end_x_point(x1, y1, x2, y2, x3, y3, d, k)
        yz = d * xz + k
        if point_is_between_two_points_on_line_segment(x1, y1, x2, y2, xz, yz):
            height_lens.append(dist(x3, y3, xz, yz))

    def line_cross_check(xa, xb, xc, xd, ya, yb, yc, yd):
        a1 = xb - xa
        a2 = yc - yd
        a3 = yb - ya
        b1 = xc - xd
        b2 = yc - ya
        b3 = xc - xa
        delta = a1 * a2 - a3 * b1
        if not delta:
            return False
        delta1 = a1 * b2 - a3 * b3
        delta2 = b3 * a2 - b2 * b1
        t = delta1 / delta
        s = delta2 / delta
        return t >= 0 and s <= 1

    (xa, ya), (xb, yb) = point_a, point_b
    (xc, yc), (xd, yd) = point_c, point_d

    if line_cross_check(xa, xb, xc, xd, ya, yb, yc, yd):
        return 0.

    height_lens = []

    slope1, shift1 = get_line_slope_shift(xa, ya, xb, yb)
    if slope1 is not None:
        add_height_len_to_list(xa, ya, xb, yb, xc, yc, shift1, slope1)
        add_height_len_to_list(xa, ya, xb, yb, xd, yd, shift1, slope1)
    else:
        slope1, shift1 = get_line_slope_shift(ya, xa, yb, xb)
        add_height_len_to_list(ya, xa, yb, xb, yc, xc, shift1, slope1)
        add_height_len_to_list(ya, xa, yb, xb, yd, xd, shift1, slope1)
    slope2, shift2 = get_line_slope_shift(xc, yc, xd, yd)
    if slope2 is not None:
        add_height_len_to_list(xc, yc, xd, yd, xa, ya, shift2, slope2)
        add_height_len_to_list(xc, yc, xd, yd, xb, yb, shift2, slope2)
    else:
        slope2, shift2 = get_line_slope_shift(yc, xc, yd, xd)
        add_height_len_to_list(yc, xc, yd, xd, ya, xa, shift2, slope2)
        add_height_len_to_list(yc, xc, yd, xd, yb, xb, shift2, slope2)
    if height_lens:
        return min(height_lens)
    else:
        return min((dist(xa, ya, xd, yd),
                    dist(xa, ya, xc, yc),
                    dist(xb, yb, xd, yd),
                    dist(xb, yb, xc, yc)))

# поиск индекса следующей цифры в bboxs в зависимости от заданного направления
def find_next_digit_idx(current_bbox_idx: int, bboxs: np.ndarray,
                        thr_distance: float, thr_angle: float,
                        direction: str, verbose=False):
    #     direction = 'left' or 'right'

    #     координаты центра текущей цифры
    _, xmin1, ymin1, xmax1, ymax1 = bboxs[current_bbox_idx]
    c_x_1, c_y_1 = (xmax1 + xmin1) / 2, (ymax1 + ymin1) / 2

    candidate_boxes = []
    if verbose: print("current box", bboxs[current_bbox_idx], "thr_distance",
                      thr_distance)

    #     1. По всем цифрам выполняем поиск:
    for i, bbox in enumerate(bboxs):

        #         1.1. Рассматриваемая цифра не подходит, переходим к следующей, если:
        #         - это текущая цифра.
        if i == current_bbox_idx:
            continue

        _, xmin2, ymin2, xmax2, ymax2 = bbox
        c_x_2, c_y_2 = (xmax2 + xmin2) / 2, (ymax2 + ymin2) / 2

        #         1.2. Рассматриваемая цифра не подходит, переходим к следующей, если:
        #         - направление поиска влево и центр рассматриваемой цифры больше текущей;
        #         - направление поиска вправо и центр рассматриваемой цифры меньше или
        #         равен текущей.
        #         if (direction == 'left' and c_x_2 > c_x_1) or (direction == 'right'
        #         and c_x_2 <= c_x_1):
        left_cond = direction == 'left' and c_x_2 >= c_x_1
        right_cond = direction == 'right' and c_x_2 <= c_x_1

        if left_cond or right_cond:
            continue

        # distance = dist(c_x_1, c_y_1, c_x_2, c_y_2)
        distance = box_line_segment_distance(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2,
                                             xmax2, ymax2, direction)
        # distance = dist(xmin1, ymin1, xmax2, ymin2) if direction == 'left' else dist(
        #     xmax1, ymin1, xmin2, ymin2)
        #         1.3. Рассматриваемая цифра не подходит, переходим к следующей, если:
        #         - расстояние между центрами текущей и рассматриваемой цифр больше
        #         порога.
        if distance > thr_distance:
            if verbose:
                print("bbox", bbox)
                print("distance", distance,
                      "c_x_1, c_y_1, c_x_2, c_y_2", c_x_1, c_y_1, c_x_2, c_y_2)
            continue

        x_3, y_3 = c_x_2, c_y_1

        angle = calc_angle_degrees(c_x_2, c_y_2, c_x_1, c_y_1, x_3, y_3)
        if verbose: print("i:", i, "bbox:", bbox, "dist:", distance, "angle:", angle)
        #         1.4. Рассматриваемая цифра не подходит, переходим к следующей, если:
        #         - угол между центрами текущей и рассматриваемой цифр больше порога.
        if angle > thr_angle:
            continue
        candidate_boxes.append((bbox, distance, box_area(bbox[1:]), i))

    if not len(candidate_boxes):
        return

    # sort by distance. inc order
    candidate_boxes = sorted(candidate_boxes, key=operator.itemgetter(1))
    first_ind = candidate_boxes[0][3]
    if direction == 'left' or len(candidate_boxes) == 1:
        return first_ind

    BOX_NUM = 2
    # tup2 - min area, tup1 - max area.
    tup1, tup2 = candidate_boxes[:BOX_NUM]
    if tup1[2] < tup2[2]:
        tup2, tup1 = tup1, tup2
    if verbose:
        print("big box:", tup1)
        print("small box:", tup2)
        print("Prev box:", first_ind, 'box', bboxs[first_ind])
    box1, dist1, area1, i1 = tup1
    box2, dist2, area2, i2 = tup2
    _, xmin1, ymin1, xmax1, ymax1 = box1
    _, xmin2, ymin2, xmax2, ymax2 = box2
    if (ymin2 > ymax1 or ymin1 > ymax2) and (xmax1 > xmin2 or xmax2 > xmin1):
        if verbose: print("if1 return:", i1, 'box', bboxs[first_ind])
        return i1
    if verbose: print("if2 return:", first_ind, 'box', bboxs[first_ind])
    return first_ind


# поиск целой части цены
def find_rub_kop_parts(bboxes: np.ndarray,
                       coeffs_rub_angle: float, coeffs_rub_area: float,
                       coeffs_rub_distance: float, coeffs_distance_x: float,
                       thr_angle=35,
                       thr_square_angle_diff=.095, thr_dist_diff=.095,
                       verbose=False):
    #     thr_angle - максимальный угол отклонения новой цифры от текущей при поиске
    #     соседних цифр
    #     thr_target_diff - порог для проверки наличия копеек в цене

    #     1. Если количество цифр меньше или равно двум, то возвращаем исходные цифры,
    #     иначе - продолжаем
    #     поиск.
    if len(bboxes) <= 2:
        return bboxes, None

    #     2. Рассчитываем:
    #     - углы между центрами соседних цифр слева направо;
    #     - отношение между площадями соседних цифр слева направо;
    #     - расстояние между центрами соседних цифр слева направо.
    center_angles = []
    squares_diffs = []
    center_dists = []
    x_diffs = []
    for k in range(len(bboxes) - 1):
        _, xmin_1, ymin_1, xmax_1, ymax_1 = bboxes[k]
        _, xmin_2, ymin_2, xmax_2, ymax_2 = bboxes[k + 1]

        c_x_1, c_y_1 = xmin_1 + (xmax_1 - xmin_1) / 2, ymin_1 + (ymax_1 - ymin_1) / 2
        c_x_2, c_y_2 = xmin_2 + (xmax_2 - xmin_2) / 2, ymin_2 + (ymax_2 - ymin_2) / 2
        x_3, y_3 = c_x_2, c_y_1

        angle = calc_angle_degrees(c_x_2, c_y_2, c_x_1, c_y_1, x_3, y_3) / thr_angle
        center_angles.append(angle)

        area1 = box_area([xmin_1, ymin_1, xmax_1, ymax_1])
        area2 = box_area([xmin_2, ymin_2, xmax_2, ymax_2])
        square_diff = area1 / area2
        squares_diffs.append(square_diff)

        c_dist = dist(c_x_1, c_y_1, c_x_2, c_y_2)
        center_dists.append(c_dist)

        width1 = xmax_1 - xmin_1
        width2 = xmax_2 - xmin_2
        max_width = max(width1, width2)
        x_diff = xmin_2 - xmax_1
        x_diff = max(x_diff / max_width, 0.)
        x_diffs.append(x_diff)
    center_angles = np.array(center_angles)
    if verbose: print('\ncenter_angles', center_angles, sum(center_angles))

    #     2.1. Отношение между площадями поэлментно делим на общую сумму всех элементов.
    if verbose: print('\nsquares_diffs', squares_diffs, sum(squares_diffs))
    squares_diffs = np.array(squares_diffs)
    # squares_diffs /= sum(squares_diffs)
    if verbose: print('squares_diffs', squares_diffs, sum(squares_diffs))

    #     2.2. Расстояние между центрами поэлментно делим на общую сумму всех элементов.
    if verbose: print('\ncenter_dists', center_dists, sum(center_dists))
    center_dists = np.array(center_dists)
    center_dists /= sum(center_dists)
    if verbose: print('center_dists', center_dists, sum(center_dists))

    if verbose: print('\nx_diffs', x_diffs, sum(x_diffs))
    x_diffs = np.array(x_diffs)
    if verbose: print('x_diffs', x_diffs, sum(x_diffs))
    #     3. Поиск интервала целой части цены и копеек.
    # todo подобрать
    target_max_values = []
    target_max_values2 = []

    #     3.1. По каждому интервалу пофрмируем целевое значение равное поэлементной сумме:
    #     (углов между центрами соседних цифр слева направо) +
    #     (отношений между площадями соседних цифр слева направо) +
    #     (расстояний между центрами соседних цифр слева направо).

    for k in range(len(bboxes) - 1):
        target_max_values.append(coeffs_rub_angle * center_angles[k] ** 2 +
                                 coeffs_rub_area * squares_diffs[k] ** 2)
        target_max_values2.append(coeffs_rub_distance * center_dists[k] ** 2 +
                                  coeffs_distance_x * x_diffs[k] ** 2)
    target_max_values = np.array(target_max_values)
    target_max_values2 = np.array(target_max_values2)

    #     3.2. Целевые значения поэлментно делим на общую сумму всех элементов.
    if verbose: print('\ntarget_max_values', target_max_values)
    target_max_values /= sum(target_max_values)
    if verbose: print('target_max_values', target_max_values)

    if verbose: print('max-mean', np.max(target_max_values) - np.mean(target_max_values))

    if verbose: print('\ntarget_max_values2', target_max_values2)
    target_max_values2 /= sum(target_max_values2)
    if verbose: print('target_max_values2', target_max_values2)

    if verbose: print('max-mean2', np.max(target_max_values2) - np.mean(
        target_max_values2))

    #     3.3. Если разница максимального и среднего целевых значений больше порога,
    #     то выделяем целую
    #     часть из цены по максимальному целевому значению.
    #     Если разница меньше порога – список цифр остается неизменным.
    thr = thr_square_angle_diff / len(target_max_values)
    thr2 = thr_dist_diff / len(target_max_values)
    if np.max(target_max_values) - np.mean(target_max_values) > thr:
        target_max_idx = np.argmax(target_max_values) + 1
        rub_bboxs = bboxes[:target_max_idx]
        kop_bboxs = bboxes[target_max_idx:target_max_idx + 2]
        return rub_bboxs, kop_bboxs
    if np.max(target_max_values2) - np.mean(target_max_values2) > thr2:
        target_max_idx = np.argmax(target_max_values2) + 1
        rub_bboxs = bboxes[:target_max_idx]
        kop_bboxs = bboxes[target_max_idx:target_max_idx + 2]
    else:
        rub_bboxs = bboxes
        kop_bboxs = None

    if verbose: print('integer price part', rub_bboxs[:, 0])
    return rub_bboxs, kop_bboxs


# поиск ценника
def find_price(bboxes: np.ndarray,
               img_h: int, img_w: int, coeffs_price_distance: float,
               coeffs_price_area: float,
               thr_angle: int = 35, thr_distance_factor: float = 2,
               verbose=False):
    #     thr_angle - максимальный угол отклонения новой цифры от текущей при поиске
    #     соседних цифр
    #     thr_distance_factor - множитель порога расстояния между цифрами

    if not len(bboxes):
        return bboxes

    #     порог расстояния между цифрами
    if verbose:
        print("thr_angle", thr_angle)
        print("thr_distance_factor", thr_distance_factor)
        print("target_coeffs", (coeffs_price_distance, coeffs_price_area))

    #     координаты нижнего правого угла (br - bottom right)
    br_x, br_y = img_w - 1, img_h - 1

    #     список цен (цена - список цифр)
    prices_bboxes = []

    #     площади цифр
    area_list = [box_area(box[1:]) for box in bboxes]

    thr_distance = thr_distance_factor * np.mean(bboxes[:, 3] - bboxes[:, 1])

    # 1. Пока есть цифры - ищем цену
    if verbose: print("Bboxes:", bboxes,
                      "widthes", bboxes[:, 3] - bboxes[:, 1])
    while len(bboxes):
        price_idxs = []

        # 1.1. Находим цифру с наименьтшм расстоянием от нижнего правого угла
        next_digit_idx = np.argmax(area_list)
        if verbose: print("Starting digit", bboxes[next_digit_idx])
        # next_box = bboxes[next_digit_idx]
        # thr_distance = thr_distance_factor * (next_box[3] - next_box[1])

        # 1.2. От текущей точки находим все цифры слева
        while next_digit_idx is not None:
            price_idxs.insert(0, next_digit_idx)
            next_digit_idx = find_next_digit_idx(next_digit_idx,
                                                 bboxes, thr_distance,
                                                 thr_angle, 'left', verbose)

        # 1.3. От точки в п1.1. находим все цифры справа
        next_digit_idx = price_idxs[-1]
        next_digit_idx = find_next_digit_idx(next_digit_idx,
                                             bboxes, thr_distance,
                                             thr_angle, 'right', verbose)
        while next_digit_idx is not None:
            price_idxs.append(next_digit_idx)
            next_digit_idx = find_next_digit_idx(next_digit_idx,
                                                 bboxes,
                                                 thr_distance,
                                                 thr_angle, 'right', verbose)

        #         1.4. Добавляем найденный список цифр (цену) в список цен
        prices_bboxes.append(np.array([bboxes[i] for i in price_idxs]))

        #         1.5. Удаляем найденные цифры из общего списка цифр и их расстояния до
        #         нижнего правого
        #         угла из списка расстояний
        bboxes = np.delete(bboxes, price_idxs, 0)
        area_list = np.delete(area_list, price_idxs, 0)

    #     2. Поиск верной цены из списка цен
    #     список целевых значений цен
    target_min_values = []
    if verbose: print("prices_bboxes", prices_bboxes)
    #     2.1. По каждой цене из списка рассчитываем поэлементно целевые значения = ((
    #     оценка расстояния) -
    #     (оценка площади))
    for bboxes in prices_bboxes:
        xmin_tag_box, ymin_tag_box, xmax_tag_box, ymax_tag_box \
            = np.min(bboxes[:, 1]), np.min(bboxes[:, 2]), \
              np.max(bboxes[:, 3]), np.max(bboxes[:, 4])
        center_tag_x, center_tag_y = (xmin_tag_box + xmax_tag_box) / 2, (ymin_tag_box +
                                                                         ymax_tag_box) / 2
        tag_to_br_dist = dist(center_tag_x, center_tag_y, br_x, br_y)
        tag_diagonal = dist(img_w, img_h, 0, 0)
        # оценка расстояния = расстояние от прямоугольника цены до правого нижнего
        # угла ценника / диагональ ценника.
        br_distance = tag_to_br_dist / tag_diagonal

        # Оценка площади = площадь бокса, составленная для цены / площадь ценника.
        tag_width = xmax_tag_box - xmin_tag_box
        tag_height = ymax_tag_box - ymin_tag_box
        area_coeff = (tag_width * tag_height) / (img_w * img_h)
        if verbose:
            print("bboxes, br_distance, area_coeff, br_distance - area_coeff:",
                  bboxes, br_distance, area_coeff, br_distance * coeffs_price_distance -
                  area_coeff * coeffs_price_area)

        target_min_values.append(br_distance * coeffs_price_distance -
                                 area_coeff * coeffs_price_area)
    #     2.2. Находим список цифр соответствующий минимальному целевому значению цен
    target_min_idx = int(np.argmin(target_min_values))
    if verbose:
        print("target_min_idx", target_min_idx)
        for price_bbox in prices_bboxes:
            print("price bbox:", price_bbox)
    bboxes = np.array(prices_bboxes[target_min_idx])
    return bboxes


# поиск ценника
def find_price_v2(bboxes: np.ndarray, img_h: int, img_w: int, trees_dict: dict,
                  thr_angle: int = 35, thr_distance_factor: float = 2,
                  not_price_class_threshold=.01, img=None, verbose=False):
    #     thr_angle - максимальный угол отклонения новой цифры от текущей при поиске
    #     соседних цифр
    #     thr_distance_factor - множитель порога расстояния между цифрами

    if not len(bboxes):
        return bboxes

    #     порог расстояния между цифрами
    if verbose:
        print("thr_angle", thr_angle)
        print("thr_distance_factor", thr_distance_factor)
    thr_distance = thr_distance_factor * np.mean(bboxes[:, 3] - bboxes[:, 1])

    #     список цен (цена - список цифр)
    prices_bboxes = []

    #     площади цифр
    area_list = [box_area(box[1:]) for box in bboxes]
    #     1. Пока есть цифры - ищем цену
    if verbose: print("Bboxes:", bboxes, "thr_distance", thr_distance,
                      "widthes", bboxes[:, 3] - bboxes[:, 1])
    # verbose = not verbose
    while len(bboxes):
        price_idxs = []

        #         1.1. Находим цифру с наименьтшм расстоянием от нижнего правого угла
        # next_digit_idx = np.argmin(br_distances)
        next_digit_idx = np.argmax(area_list)
        if verbose: print("Starting digit", bboxes[next_digit_idx])

        #         1.2. От текущей точки находим все цифры слева
        while next_digit_idx is not None:
            price_idxs.insert(0, next_digit_idx)
            next_digit_idx = find_next_digit_idx(next_digit_idx,
                                                 bboxes, thr_distance,
                                                 thr_angle, 'left', verbose)

        #         1.3. От точки в п1.1. находим все цифры справа
        next_digit_idx = price_idxs[-1]
        next_digit_idx = find_next_digit_idx(next_digit_idx,
                                             bboxes, thr_distance,
                                             thr_angle, 'right', verbose)
        while next_digit_idx is not None:
            price_idxs.append(next_digit_idx)
            next_digit_idx = find_next_digit_idx(next_digit_idx,
                                                 bboxes,
                                                 thr_distance,
                                                 thr_angle, 'right', verbose)
        #         1.4. Добавляем найденный список цифр (цену) в список цен
        prices_bboxes.append([bboxes[i] for i in price_idxs])

        #         1.5. Удаляем найденные цифры из общего списка цифр и их расстояния до
        #         нижнего правого
        #         угла из списка расстояний
        bboxes = np.delete(bboxes, price_idxs, 0)
        area_list = np.delete(area_list, price_idxs, 0)

    prices_bboxes = np.array([np.array([np.array([z for z in y]) for y in x]) for x in
                              prices_bboxes])
    res = []
    if img is not None: print("len price bboxes", len(prices_bboxes))
    MAX_TAG_NUM = 3
    for price_boxes_i in prices_bboxes[:MAX_TAG_NUM]:
        boxes_list_len = len(price_boxes_i)
        if boxes_list_len == 1:
            # res.append((price_boxes_i, price_boxes_i[1:]))
            continue
        if boxes_list_len > 5:
            continue

        pred, pred_proba = tree_predict(boxes_list_len, img_h, img_w, price_boxes_i,
                                        trees_dict, False)
        if img is not None:
            print("pred, pred_proba,", pred, pred_proba, "flag",
                  pred_proba[-1] > not_price_class_threshold)
            tree_predict(boxes_list_len, img_h, img_w, price_boxes_i,
                         trees_dict, True, debug=True)
        if (pred == (boxes_list_len + 1)) and (
                pred_proba[-1] >= not_price_class_threshold):
            # add_vector = np.zeros(boxes_list_len * 4, 'float32')
            # for i in range(2, boxes_list_len * 4, 4):
            #     add_vector[i] = -.03
            #     add_vector[i] = np.random.uniform(-.0005, .0005)
            #     add_vector[i + 1] = np.random.uniform(-.0005, .0005)
            # tree_predict(boxes_list_len, img_h, img_w, price_boxes_i,
            #              trees_dict, add_vector, True)
            continue
        rub_boxes = price_boxes_i[:pred]
        kop_boxes = price_boxes_i[pred:]
        if img is not None:
            print("(pred != (boxes_list_len + 1)", (pred != (boxes_list_len + 1)))
            print("pred_proba[-1] > not_price_class_threshold",
                  pred_proba[-1] > not_price_class_threshold)
            print("pred == (boxes_list_len + 1)", pred == (boxes_list_len + 1))
            print("pred_proba[-1]", pred_proba[-1])
            print("pred, pred_proba", pred, pred_proba,
                  len(price_boxes_i), boxes_list_len, price_boxes_i, "r", rub_boxes,
                  "k", kop_boxes)
        res.append((rub_boxes, kop_boxes))

    return res


def tree_predict(boxes_len, img_h, img_w, boxes, trees_dict, add_vector=False,
                 debug=False):
    cur_tree: DecisionTreeClassifier = trees_dict[boxes_len]
    boxes = boxes.copy()
    min_max_boxes_to_center_boxes(boxes[:, 1:])
    cur_features = np.concatenate(tuple(boxes[:, 1:])).astype(
        'float32')
    normalize_features(img_h, img_w, cur_features)
    if add_vector:
        add_vector = np.zeros(boxes_len * 4, 'float32')
        add_vector[6] = -0.
        # for i in range(2, boxes_len * 4, 4):
        #     x_tr = - .2 * cur_features[i]
        #     if cur_features[i] + x_tr > .05:
        #         add_vector[i] = x_tr
        # if debug:
        #     print("add_vector", add_vector, "cur_features", cur_features)
        cur_features += add_vector
    x_pred = [cur_features]
    if debug: print(x_pred)
    pred = cur_tree.predict(x_pred)
    pred_proba = cur_tree.predict_proba(x_pred)
    if debug:
        print(f"features shape: {cur_features.shape[0] // 4},"
              f" pred: {pred}, pred_proba: {pred_proba}")
    return pred[0], pred_proba[0]
