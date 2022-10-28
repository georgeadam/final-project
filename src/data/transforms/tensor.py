import torch

from .creation import transforms


class Tensor:
    def __call__(self, arr):
        return torch.from_numpy(arr)


transforms.register_builder("tensor", Tensor)
