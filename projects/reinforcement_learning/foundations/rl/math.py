from typing import Callable

import numpy as np


def get_logistic_function(alpha: float) -> Callable[[float], float]:
    return lambda x: 1.0 / (1.0 + np.exp(-alpha * x))


def get_unit_sigmoid_function(alpha: float) -> Callable[[float], float]:
    return lambda x: 1.0 / (1.0 + (1.0 / np.where(x == 0, 1e-8, x) - 1) ** alpha)
