from structure import get_structure
from pprint import pprint
from inspect import signature
import torch
import numpy as np
from typing import Callable
import cv2
import random


class Cascade:
    """
    Базовый класс для всех Каскадов

    - Есть имя, которое можно задать
    - Его строковое представление - это его имя (__str__ и __repr__ переопределены для логирования)
    - Этот объект вызываем (обязательно для всех функций и моделей)
    """
    name: str = "Каскад"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class CascadeElement(Cascade):
    """
    Используются для вызова функции или модели
    """

    def __init__(self, fun: Callable, name: str = None):
        self.fun = fun
        self.signature = signature(fun)
        self.name = name if name is not None else self.name

    def __call__(self, *agr):
        self.out = self.fun(*agr)
        return self.out


class CascadeBlock(Cascade):
    """
    Занимается всей логикой для связи каскадов
    Может содержать CascadeElement и CascadeBlock
    Необходим list из Каскадов и матрица смежностей их
    (dict где ключ - id Каскада в list, а значение - id Каскадов, выход из которых пойдёт на вход)

    Если при вызове Каскад передал None, то передаёт None (важно учитывать при создании функций)
    Имя блока - это все имена каскадов, из которых он состоит (у заготовленных блоков __str__ - это заготовленное имя)
    """

    def __init__(self, cascades_list: list, adjacency_map: dict):

        self.cascades_list = cascades_list
        self.adjacency_map = adjacency_map
        self.name = "[" + ", ".join([str(x.name) for x in self.cascades_list]) + "]"

    # Лучше продумать, переделать
    def __call__(self, item):
        global i
        self.out_map = {}
        for i, cascade in enumerate(self.cascades_list):
            self.out_map[i] = cascade(
                *[item if j == 'ITER' else self.out_map[j] for j in self.adjacency_map[i]]
            )

            if self.out_map[i] is None:
                return None

        return self.out_map[i]

    def loop(self, iterator):
        for item in iterator:
            yield self.__call__(item)


struct = get_structure()
pprint(struct)

count = struct.function.object_detection.count

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')


def parse_model(img: np.ndarray) -> np.ndarray:
    out = model(img)
    out = out.pred[0].numpy()
    return out[out[:, -1] == 0]  # 0 is person


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


def plot_all_bbox(bbox: np.ndarray, img: np.ndarray):
    for *xyxy, id in bbox:
        plot_box(xyxy, img, label=str(id))
    print(img.shape)  # это функция работает только в коллабе


sort = struct.model.tracking.sort()

num = np.ones((1, 3))

count_element = CascadeElement(count, name="Подсчёт")
model_cas = CascadeElement(parse_model, name="Yolo модель")
print_cas = CascadeElement(plot_all_bbox, name="принтушка")

print(count_element.signature)
print(len(count_element.signature.parameters))
print(count_element.signature.return_annotation)
# print(model_cas.signature.return_annotation)

