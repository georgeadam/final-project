import numpy as np
import torch
from skimage.filters import gaussian

from .applicator import Applicator
from .creation import applicators


class GaussianBlur(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [.4, .6, 0.7, .8, 1][self.severity - 1]

        x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
        return np.clip(x, 0, 1) * 255


applicators.register_builder("gaussian_blur", GaussianBlur)
