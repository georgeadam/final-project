import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class ErrorOscillation(LabelCorruptor):
    def __init__(self, corruption_prob, sample_limit):
        self.corruption_prob = corruption_prob
        self.sample_limit = sample_limit

    def corrupt_helper(self, preds, y):
        y = copy.deepcopy(y)
        indices = self.get_corruption_indices(preds, y)
        y[indices] = 0

        return y

    def get_corruption_indices(self, preds, y):
        indices = self.get_relevant_indices(preds, y)
        indices = np.random.choice(indices, size=int(self.corruption_prob * len(indices)), replace=False)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, preds, y):
        return np.where(np.logical_and(y == 1, preds == 0))[0]


label_corruptors.register_builder("error_oscillation", ErrorOscillation)
