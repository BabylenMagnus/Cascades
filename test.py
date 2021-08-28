from pydantic_structer.object_detection.yolo import ParametersExtraData
from pydantic_structer.mixins import BaseMixinData
# from keras.models import load_model
from cascades.object_detection.yolo import YoloCascade
from inspect import signature
from pydantic import BaseModel
from typing import Any
from pydantic import validator
from typing import Callable

#
# model = load_model('yolo_head.h5')
# params_dict = dict(model=lambda x: 2,
#                    frame_size=416,
#                    score_threshold=.2,
#                    iou_threshold=.7,
#                    soft_nms=True,
#                    sigma=.9
#                    )
#
# params = ParametersExtraData(**params_dict)

# print(params.dict())


class A(BaseMixinData):
    name: str = "Name"
    source: Any

    @validator('source', always=True)
    def _validate_check_source(cls, value):
        if not isinstance(value, Callable):
            raise Exception("Ошибка")
        return value

    def dict(self, **kwargs):
        kwargs.update({'exclude': {'source'}})
        return super().dict(**kwargs)


# print(A(source=lambda x: 4).dict())


# a = type(BaseModel, (), {**params_dict})
# a = a()
# print(a.dict())
# class NewBaseMixinData(BaseModel):
#     def __init__(self, **data):
#         for __name, __field in self.__fields__.items():
#             __type = __field.type_
#             if hasattr(__type, "__mro__") and UniqueListMixin in __type.__mro__:
#                 data.update({__name: __type(data.get(__name, __type()))})
#         super().__init__(**data)


# for i in params.__fields__.items():
#     print(i)
#
# print('\n\n')
#

# for ty in signature(YoloCascade).parameters.items():
#     print(ty)

print(signature(YoloCascade).parameters)
# print(signature(YoloCascade).parameters)

# print(params)

# print(YoloCascade(model))
# print(YoloCascade(**params))
