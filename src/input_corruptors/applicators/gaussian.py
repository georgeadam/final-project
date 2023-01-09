import numpy as np
import torch

from .applicator import Applicator
from .creation import applicators


class Gaussian(Applicator):
    def apply_corruption(self, x):
        x = [self.corrupt_single_sample(x[i]) for i in range(len(x))]

        return np.array(x)

    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [0.04, 0.06, .08, .09, .10][self.severity - 1]
        x = np.array(x) / 255.

        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


applicators.register_builder("gaussian", Gaussian)
