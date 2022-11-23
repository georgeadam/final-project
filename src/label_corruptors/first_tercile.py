import copy
import os

import numpy as np
import pandas as pd

from settings import ROOT_DIR
from .creation import label_corruptors
from .difficulty_corruptor import DifficultyCorruptor


class FirstTercile(DifficultyCorruptor):
    def __init__(self, counts_path, sample_limit):
        super().__init__(sample_limit)
        self.counts = pd.read_csv(os.path.join(ROOT_DIR, counts_path))

    def corrupt_helper(self, preds, y, indices, **kwargs):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, indices)
        y[corruption_indices] = 1 - y[corruption_indices]

        return y

    def get_relevant_indices(self, sample_indices, **kwargs):
        return sample_indices

    def get_difficult_indices(self):
        upper_bound = np.percentile(self.counts["correct"], 33)

        return self.counts.loc[self.counts["correct"] < upper_bound]["sample_idx"].to_numpy()


label_corruptors.register_builder("first_tercile", FirstTercile)