import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class ErrorAmplification(LabelCorruptor):
    def __init__(self, corruption_prob, sample_limit):
        super().__init__(sample_limit)
        self.corruption_prob = corruption_prob

    def corrupt_helper(self, preds, y):
        y = copy.deepcopy(y)
        indices = self.get_corruption_indices(preds, y)
        y[indices] = 1

        return y

    def get_corruption_indices(self, preds, y):
        indices = self.get_relevant_indices(preds, y)
        indices = np.random.choice(indices, size=int(self.corruption_prob * len(indices)), replace=False)
        indices = self.subset_indices(indices, self.sample_limit)

        return indices

    def get_relevant_indices(self, preds, y):
        return np.where(np.logical_and(y == 0, preds == 1))[0]


label_corruptors.register_builder("error_amplification", ErrorAmplification)
