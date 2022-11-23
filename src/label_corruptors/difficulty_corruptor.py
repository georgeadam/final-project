import abc

import numpy as np

from .label_corruptor import LabelCorruptor


class DifficultyCorruptor(LabelCorruptor):
    @abc.abstractmethod
    def get_difficult_indices(self):
        raise NotImplementedError

    def get_actual_indices(self, preds, sample_indices, **kwargs):
        corruption_indices = self.get_corruption_indices(preds, sample_indices)

        return sample_indices[corruption_indices]

    def get_potential_indices(self, preds, sample_indices, **kwargs):
        relevant_indices = self.get_relevant_indices(preds=preds, sample_indices=sample_indices)
        difficult_indices = self.get_difficult_indices()

        potential_indices = np.intersect1d(relevant_indices, difficult_indices)

        return potential_indices

    def get_corruption_indices(self, preds, sample_indices):
        relevant_indices = self.get_relevant_indices(preds=preds, sample_indices=sample_indices)
        difficult_indices = self.get_difficult_indices()

        corruption_indices = np.intersect1d(relevant_indices, difficult_indices)
        corruption_indices = self.subset_indices(corruption_indices, self.sample_limit)

        sorted_sample_indices = sample_indices.argsort()
        corruption_indices = sorted_sample_indices[
            np.searchsorted(sample_indices, corruption_indices, sorter=sorted_sample_indices)]

        return corruption_indices
