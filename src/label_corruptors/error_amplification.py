import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class ErrorAmplification(LabelCorruptor):
    def __init__(self, corruption_prob, noise_tracker, num_classes, sample_limit, seed):
        super().__init__(noise_tracker, num_classes, sample_limit, seed)
        self.corruption_prob = corruption_prob

    def corrupt_helper(self, preds, y, **kwargs):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, y)
        y[corruption_indices] = 1

        return y[corruption_indices], corruption_indices

    def get_actual_indices(self, preds, y, sample_indices, **kwargs):
        corruption_indices = self.get_corruption_indices(preds, y)

        return list(sample_indices[corruption_indices])

    def get_potential_indices(self, preds, y, sample_indices, **kwargs):
        potential_indices = self.get_relevant_indices(preds, y)

        return list(sample_indices[potential_indices])

    def get_corruption_indices(self, preds, y):
        indices = self.get_relevant_indices(preds, y)
        random_state = np.random.RandomState(self.seed)
        indices = random_state.choice(indices, size=int(self.corruption_prob * len(indices)), replace=False)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, preds, y):
        return np.where(np.logical_and(y == 0, preds == 1))[0]


label_corruptors.register_builder("error_amplification", ErrorAmplification)
