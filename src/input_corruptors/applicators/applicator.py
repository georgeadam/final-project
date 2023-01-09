import abc
import numpy as np
from torchvision import transforms


class Applicator:
    def __init__(self, severity):
        self.severity = severity
        self.transform = transforms.ToPILImage()

    def apply_corruption(self, x):
        x = [self.corrupt_single_sample(x[i]) for i in range(len(x))]

        return np.array(x)

    @abc.abstractmethod
    def corrupt_single_sample(self, x):
        raise NotImplementedError