from typing import Callable
from inspect import signature


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


class PreMadeCascade(CascadeBlock):
    def __str__(self):
        return self.name

    def __repr__(self):
        return "[" + ", ".join([str(x.name) for x in self.cascades_list]) + "]"
