import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class NegativeBiasNoise(LabelCorruptor):
    def __init__(self, noise_level, sample_limit, seed):
        super().__init__(sample_limit, seed)
        self.noise_level = noise_level

    def corrupt_helper(self, preds, y, **kwargs):
        y = copy.deepcopy(y)
        indices = self.get_corruption_indices(preds)
        y[indices] = 0

        return y

    def get_actual_indices(self, preds, sample_indices, **kwargs):
        corruption_indices = self.get_corruption_indices(preds)

        return sample_indices[corruption_indices]

    def get_potential_indices(self, preds, sample_indices, **kwargs):
        potential_indices = self.get_relevant_indices(preds)

        return sample_indices[potential_indices]

    def get_corruption_indices(self, preds):
        indices = self.get_relevant_indices(preds)
        random_state = np.random.RandomState(self.seed)
        indices = random_state.choice(indices, size=int(self.noise_level * len(indices)), replace=False)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, preds):
        return np.where(preds == 1)[0]


label_corruptors.register_builder("negative_bias_noise", NegativeBiasNoise)
