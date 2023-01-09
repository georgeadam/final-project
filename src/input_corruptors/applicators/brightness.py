import numpy as np
import skimage as sk
import torch

from .applicator import Applicator
from .creation import applicators


class Brightness(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [.05, .1, .15, .2, .3][self.severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1) * 255


applicators.register_builder("brightness", Brightness)
