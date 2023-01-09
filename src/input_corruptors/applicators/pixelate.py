import numpy as np
import torch
from PIL import Image as PILImage

from .applicator import Applicator
from .creation import applicators


class Pixelate(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [0.95, 0.9, 0.85, 0.75, 0.65][self.severity - 1]

        x = x.resize((int(32 * c), int(32 * c)), PILImage.BOX)
        x = x.resize((32, 32), PILImage.BOX)
        x = np.array(x)

        return x


applicators.register_builder("pixelate", Pixelate)
