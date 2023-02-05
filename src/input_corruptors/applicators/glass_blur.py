import numpy as np
import torch
from skimage.filters import gaussian

from .applicator import Applicator
from .creation import applicators


class GlassBlur(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][self.severity - 1]

        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
        height, width = x.shape[0], x.shape[1]

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(height - c[1], c[1], -1):
                for w in range(width - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


applicators.register_builder("glass_blur", GlassBlur)
