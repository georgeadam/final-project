import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class Unaligned(LabelCorruptor):
    def __init__(self, noise_tracker, sample_limit, seed):
        super().__init__(noise_tracker, sample_limit, seed)

    def corrupt_helper(self, preds, y, **kwargs):
        y = copy.deepcopy(y)
        indices = self.get_corruption_indices(preds, y)
        y[indices] = 1 - y[indices]

        return y

    def get_actual_indices(self, preds, y, sample_indices, **kwargs):
        corruption_indices = self.get_corruption_indices(preds, y)

        return list(sample_indices[corruption_indices])

    def get_potential_indices(self, preds, y, sample_indices, **kwargs):
        potential_indices = self.get_relevant_indices(preds, y)

        return list(sample_indices[potential_indices])

    def get_corruption_indices(self, preds, y):
        indices = self.get_relevant_indices(preds, y)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, preds, y):
        return np.where(np.logical_and(y == 1, preds == 1) | np.logical_and(y == 0, preds == 0))[0]


label_corruptors.register_builder("unaligned", Unaligned)
