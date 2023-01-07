import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class Aligned(LabelCorruptor):
    def __init__(self, noise_tracker, num_classes, sample_limit, seed):
        super().__init__(noise_tracker, num_classes, sample_limit, seed)

    def corrupt_helper(self, preds, y, **kwargs):
        y = copy.deepcopy(y)
        indices = self.get_corruption_indices(preds, y)
        y[indices] = preds[indices]

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
        return np.where(y != preds)[0]


label_corruptors.register_builder("aligned", Aligned)
