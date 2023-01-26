import numpy as np
import torch

from .applicator import Applicator
from .creation import applicators


class Clean(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)

        return np.array(x)


applicators.register_builder("clean", Clean)
