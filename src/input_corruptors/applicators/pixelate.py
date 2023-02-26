import numpy as np
import torch
from PIL import Image as PILImage

from .applicator import Applicator
from .creation import applicators


class Pixelate(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        shape = x.shape[1:]
        x = self.transform(x)
        c = [0.95, 0.9, 0.85, 0.75, 0.65][self.severity - 1]

        x = x.resize((int(shape[0] * c), int(shape[1] * c)), PILImage.BOX)
        x = x.resize((shape[0], shape[1]), PILImage.BOX)
        x = np.array(x)

        return x


applicators.register_builder("pixelate", Pixelate)
