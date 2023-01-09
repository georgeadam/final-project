import numpy as np
import skimage as sk
import torch

from .applicator import Applicator
from .creation import applicators


class ImpulseNoise(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)
        c = [.01, .02, .03, .05, .07][self.severity - 1]

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255


applicators.register_builder("impulse_noise", ImpulseNoise)
