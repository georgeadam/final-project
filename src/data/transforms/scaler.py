import copy

import torch

from .creation import transforms


class Scaler:
    def __init__(self, cols):
        super(Scaler, self).__init__()
        self.cols = cols
        self._mean = None
        self._std = None

    def fit(self, x):
        if self.cols is not None:
            self._mean = torch.mean(x[:, self.cols], dim=0)
            self._std = torch.std(x[:, self.cols], dim=0)
        else:
            self._mean = torch.mean(x, dim=0)
            self._std = torch.std(x, dim=0)

    def __call__(self, x):
        x = copy.deepcopy(x)

        if self.cols:
            x[self.cols] = (x[self.cols] - self._mean) / self._std
        else:
            x = (x - self._mean) / self._std

        return x


transforms.register_builder("scaler", Scaler)
