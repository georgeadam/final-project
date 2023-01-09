import os

import numpy as np
import pandas as pd

from settings import ROOT_DIR
from .creation import input_corruptors
from .input_corruptor import InputCorruptor


class FirstTercile(InputCorruptor):
    def __init__(self, counts_path, noise_tracker, applicator_args, sample_limit, seed):
        super().__init__(noise_tracker, applicator_args, sample_limit, seed)
        self.counts = pd.read_csv(os.path.join(ROOT_DIR, counts_path))

    def get_relevant_indices(self, sample_indices, **kwargs):
        return sample_indices

    def get_difficult_indices(self):
        upper_bound = np.percentile(self.counts["correct"], 33)

        return self.counts.loc[self.counts["correct"] < upper_bound]["sample_idx"].to_numpy()


input_corruptors.register_builder("first_tercile", FirstTercile)
