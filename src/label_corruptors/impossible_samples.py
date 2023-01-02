import copy
import os

import pandas as pd

from settings import ROOT_DIR
from .creation import label_corruptors
from .difficulty_corruptor import DifficultyCorruptor
from .utils import generate_multiclass_noisy_labels


class ImpossibleSamples(DifficultyCorruptor):
    def __init__(self, counts_path, noise_tracker, num_classes, sample_limit, seed):
        super().__init__(noise_tracker, num_classes, sample_limit, seed)
        self.counts = pd.read_csv(os.path.join(ROOT_DIR, counts_path))

    def corrupt_helper(self, preds, y, sample_indices, **kwargs):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, sample_indices)
        y[corruption_indices] = generate_multiclass_noisy_labels(y[corruption_indices], self.num_classes, self.seed)

        return y

    def get_relevant_indices(self, sample_indices, **kwargs):
        return sample_indices

    def get_difficult_indices(self):
        return self.counts.loc[self.counts["correct"] == self.counts["correct"].min()]["sample_idx"].to_numpy()


label_corruptors.register_builder("impossible_samples", ImpossibleSamples)
