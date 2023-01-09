from io import BytesIO

import numpy as np
import torch
from PIL import Image as PILImage

from .applicator import Applicator
from .creation import applicators


class Jpeg(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [80, 65, 58, 50, 40][self.severity - 1]

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = PILImage.open(output)
        x = np.array(x)

        return x


applicators.register_builder("jpeg", Jpeg)
