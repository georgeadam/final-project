import copy
import os

import numpy as np
import pandas as pd

from settings import ROOT_DIR
from .creation import label_corruptors
from .difficulty_corruptor import DifficultyCorruptor


class ImpossibleNegativeSamples(DifficultyCorruptor):
    def __init__(self, counts_path, sample_limit):
        super().__init__(sample_limit)
        self.counts = pd.read_csv(os.path.join(ROOT_DIR, counts_path))

    def corrupt_helper(self, preds, y, indices, **kwargs):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, indices)
        y[corruption_indices] = 1

        return y

    def get_relevant_indices(self, preds, sample_indices):
        indices = np.where(preds == 0)[0]
        indices = sample_indices[indices]

        return indices

    def get_difficult_indices(self):
        return self.counts.loc[self.counts["correct"] == self.counts["correct"].min()]["sample_idx"].to_numpy()


label_corruptors.register_builder("impossible_negative_samples", ImpossibleNegativeSamples)
