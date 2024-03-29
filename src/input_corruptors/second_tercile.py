import os

import numpy as np
import pandas as pd

from settings import ROOT_DIR
from .creation import input_corruptors
from .difficulty_corruptor import DifficultyCorruptor


class SecondTercile(DifficultyCorruptor):
    def __init__(self, counts_path, noise_tracker, applicator_args, sample_limit, seed):
        super().__init__(noise_tracker, applicator_args, sample_limit, seed)
        self.counts = pd.read_csv(os.path.join(ROOT_DIR, counts_path))

    def get_relevant_indices(self, sample_indices, **kwargs):
        return sample_indices

    def get_difficult_indices(self):
        lower_bound = np.percentile(self.counts["correct"], 33)
        upper_bound = np.percentile(self.counts["correct"], 66)

        return self.counts.loc[(self.counts["correct"] >= lower_bound) & (self.counts["correct"] < upper_bound)]["sample_idx"].to_numpy()


input_corruptors.register_builder("second_tercile", SecondTercile)
