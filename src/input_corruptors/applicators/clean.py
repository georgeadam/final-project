from .applicator import Applicator
from .creation import applicators

import torch


class Clean(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)

        return x


applicators.register_builder("clean", Clean)
