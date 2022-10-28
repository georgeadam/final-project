import copy

import numpy as np

from .creation import transforms


class Scaler:
    def __init__(self, cols):
        super(Scaler, self).__init__()
        self.cols = cols
        self._mean = None
        self._std = None

    def fit(self, x):
        if self.cols is not None:
            self._mean = np.mean(x[:, self.cols], axis=0)
            self._std = np.std(x[:, self.cols], axis=0)
        else:
            self._mean = np.mean(x, axis=0)
            self._std = np.std(x, axis=0)

    def __call__(self, x):
        x = copy.deepcopy(x)

        if self.cols:
            x[self.cols] = (x[self.cols] - self._mean) / self._std
        else:
            x = (x - self._mean) / self._std

        return x


transforms.register_builder("scaler", Scaler)
