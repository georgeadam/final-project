import copy
import os

import numpy as np
import pandas as pd

from settings import ROOT_DIR
from .creation import label_corruptors
from .difficulty_corruptor import DifficultyCorruptor


class ImpossiblePositiveSamples(DifficultyCorruptor):
    def __init__(self, counts_path, noise_tracker, num_classes, sample_limit, seed):
        super().__init__(noise_tracker, num_classes, sample_limit, seed)
        self.counts = pd.read_csv(os.path.join(ROOT_DIR, counts_path))

    def corrupt_helper(self, preds, y, sample_indices, **kwargs):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, sample_indices)
        y[corruption_indices] = 0

        return y[corruption_indices], corruption_indices

    def get_relevant_indices(self, preds, sample_indices):
        indices = np.where(preds == 1)[0]
        indices = sample_indices[indices]

        return indices

    def get_difficult_indices(self):
        return self.counts.loc[self.counts["correct"] == self.counts["correct"].min()]["sample_idx"].to_numpy()


label_corruptors.register_builder("impossible_positive_samples", ImpossiblePositiveSamples)
