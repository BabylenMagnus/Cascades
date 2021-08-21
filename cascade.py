class Cascade:
    name: str = "Каскад"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        self.out = self.fun(*args)
        return self.out


class CascadeElement(Cascade):
    def __init__(self, fun, name=None):
        self.fun = fun
        self.name = name if name is not None else self.name


class CascadeBlock(Cascade):
    def __init__(self, cascades_list: list, adjacency_map: dict):

        self.cascades_list = cascades_list
        self.adjacency_map = adjacency_map
        self.name = "[" + ", ".join([str(x.name) for x in self.cascades_list]) + "]"

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


class PreMadeCascade(CascadeBlock):
    def __str__(self):
        return self.name

    def __repr__(self):
        return "[" + ", ".join([str(x.name) for x in self.cascades_list]) + "]"
