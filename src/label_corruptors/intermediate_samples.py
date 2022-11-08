import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class IntermediateSamples(LabelCorruptor):
    def __init__(self, counts, sample_limit):
        super().__init__(sample_limit)
        self.counts = counts

    def corrupt_helper(self, preds, y, indices):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, indices)
        y[corruption_indices] = 1 - y[corruption_indices]

        return y

    def get_relevant_indices(self, sample_indices, **kwargs):
        return sample_indices

    def get_difficult_indices(self):
        return self.counts.loc[(self.counts["correct"] > self.counts["correct"].min()) & (
                self.counts["correct"] < self.counts["correct"].max())]["sample_idx"].to_numpy()


label_corruptors.register_builder("intermediate_samples", IntermediateSamples)
