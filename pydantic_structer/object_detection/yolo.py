from ..mixins import BaseMixinData
from typing import Callable
from pydantic import PositiveInt
from pydantic.types import confloat
from pydantic import BaseModel


ConstrainedFloatValueGe0Le1 = confloat(ge=0, le=1)


class ParametersExtraData(BaseMixinData):
    model: Callable  # йоло создаётся в другом месте
    frame_size: PositiveInt = 416
    score_threshold: ConstrainedFloatValueGe0Le1 = .3
    iou_threshold: ConstrainedFloatValueGe0Le1 = .45
    soft_nms: bool = False
    sigma: ConstrainedFloatValueGe0Le1 = .31
