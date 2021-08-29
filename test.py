from keras.models import load_model
from structure import get_structure
from cascade import CascadeElement

import cv2
import random
import numpy as np
from collections import OrderedDict


def plot_box(x, img, label=None):
    # Plots one bounding box on image img
    tl = 2
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = 2
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 1, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_all_bbox(bbox: np.ndarray, img: np.ndarray) -> np.ndarray:
    for *xyxy, id in bbox:
        plot_box(xyxy, img, label=str(id))
    print(img.shape)  # это функция работает только в коллабе
    return img


def validate(cascades_list, adjacency_map):
    for key, inp in adjacency_map.items():
        element = cascades_list[key]
        print(element.input, element.output)
    print('\n\n')
    pass


struct = get_structure()
class_yolo = struct.model.object_detection.yolo
class_sort = struct.model.tracking.sort
model = load_model('yolo_head.h5')

yolo = class_yolo(model)
sort = class_sort()
plot = CascadeElement(plot_all_bbox, name="Рисовалка")

adjacency_map = OrderedDict([
    (yolo, ['ITER']),
    (sort, [yolo]),
    (plot, [sort, 'ITER'])
])

item = None

a = [item if j == 'ITER' else j.out for j in adjacency_map[sort]]

print(adjacency_map)
# print([x for x in adjacency_map])
# for cascade, inp in adjacency_map.items():
#     print(cascade, inp)


#
# # print(validate(
#     [yolo, sort, plot],
#     {
#         0: ['ITER'],
#         1: [0],
#         2: [1, 'ITER']
#     }
# ))
# #
# print(validate(
#     [yolo, sort, plot],
#     {
#         0: ['ITER'],
#         1: [0, 'ITER'],
#         2: [1, 'ITER']
#     }
# ))
#
# print(validate(
#     [sort, yolo, plot],
#     {
#         0: ['ITER'],
#         1: [0],
#         2: [1, 'ITER']
#     }
# ))
