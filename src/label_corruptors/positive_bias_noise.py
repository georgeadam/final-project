import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class PositiveBiasNoise(LabelCorruptor):
    def __init__(self, noise_level, sample_limit):
        super().__init__(sample_limit)
        self.noise_level = noise_level

    def corrupt_helper(self, preds, y, **kwargs):
        y = copy.deepcopy(y)
        indices = self.get_corruption_indices(preds)
        y[indices] = 1

        return y

    def get_corruption_indices(self, preds):
        indices = self.get_relevant_indices(preds)
        indices = np.random.choice(indices, size=int(self.noise_level * len(indices)), replace=False)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, preds):
        return np.where(preds == 0)[0]


label_corruptors.register_builder("positive_bias_noise", PositiveBiasNoise)
