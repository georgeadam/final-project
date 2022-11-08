import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class UniformNoise(LabelCorruptor):
    def __init__(self, noise_level, sample_limit):
        super().__init__(sample_limit)
        self.noise_level = noise_level

    def corrupt_helper(self, preds, y):
        y = copy.deepcopy(y)
        indices = self.get_corruption_indices(y)
        y[indices] = 1 - y[indices]

        return y

    def get_corruption_indices(self, y):
        indices = self.get_relevant_indices(y)
        indices = np.random.choice(indices, size=int(self.noise_level * len(indices)), replace=False)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, y):
        return np.arange(len(y))


label_corruptors.register_builder("uniform_noise", UniformNoise)
