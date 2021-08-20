from tensorflow.keras.models import load_model
from function import *


class Cascade:
    name: str = "Cascade"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        if 'model' in self.__dict__:
            self.out = self.model(*args)
            return self.out
        if 'fun' in self.__dict__:
            self.out = self.fun(*args)
            return self.out


class CascadeFunction(Cascade):
    def __init__(self, fun, name=None):
        self.fun = fun
        self.name = name if name is not None else self.name


class CascadeModel(Cascade):
    def __init__(self, weight, name=None):
        self.model = load_model(weight)
        self.name = name if name is not None else self.name


class CascadeBlock(Cascade):
    def __init__(self, cascades_list: list, adjacency_map: dict):

        self.cascades_list = cascades_list
        self.adjacency_map = adjacency_map

    def __repr__(self):
        return "Каскад из:\n" + ", ".join([str(x.name) for x in self.cascades_list])

    def __str__(self):
        return "Каскад из:\n" + ", ".join([str(x.name) for x in self.cascades_list])

    # Лучше продумать, переделать
    def __call__(self, item):
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


class YoloCascade(CascadeBlock):

    def __init__(
            self, weight, frame_size=416, score_threshold=.3, iou_threshold=.45, method='nms', sigma=0.3, name=None
    ):
        self.preprocess_cascad = CascadeFunction(
            preprocess_video(frame_size),
            "Препроцесс"
        )

        self.yolo = CascadeModel(weight, "Yolo")

        self.postprocess_cascad = CascadeFunction(
            postprocess_yolo(frame_size, score_threshold, iou_threshold, method, sigma),
            "Постобработка"
        )

        cascades_list = [
            self.preprocess_cascad, self.yolo, self.postprocess_cascad
        ]

        adjacency_map = {
            0: ['ITER'],
            1: [0],
            2: [1, 'ITER']
        }

        super().__init__(cascades_list, adjacency_map)

        self.name = name if name is not None else "Yolo Каскад"

    def get_loop_out(self, iterator):
        return [x for x in self.loop(iterator)]


class CascadeDataset:
    pass
