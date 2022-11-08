import copy

import numpy as np

from .creation import label_corruptors
from .difficulty_corruptor import DifficultyCorruptor


class EasyNegativeSamples(DifficultyCorruptor):
    def __init__(self, counts, sample_limit):
        super().__init__(sample_limit)
        self.counts = counts

    def corrupt_helper(self, preds, y, indices):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, indices)
        y[corruption_indices] = 1

        return y

    def get_relevant_indices(self, preds, sample_indices):
        indices = np.where(preds == 0)[0]
        indices = sample_indices[indices]

        return indices

    def get_difficult_indices(self):
        return self.counts.loc[self.counts["correct"] == self.counts["correct"].max()]["sample_idx"].to_numpy()


label_corruptors.register_builder("easy_negative_samples", EasyNegativeSamples)
