import numpy as np

from .creation import transforms


class Transpose:
    def __call__(self, arr):
        return np.transpose(arr, (1, 2, 0))


transforms.register_builder("transpose", Transpose)
