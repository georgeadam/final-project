import numpy as np
import torch

from .applicator import Applicator
from .creation import applicators


class Contrast(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [.75, .5, .4, .3, 0.15][self.severity - 1]

        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * c + means, 0, 1) * 255


applicators.register_builder("contrast", Contrast)
