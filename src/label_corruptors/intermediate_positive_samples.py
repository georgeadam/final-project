import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class IntermediatePositiveSamples(LabelCorruptor):
    def __init__(self, counts, sample_limit):
        self.counts = counts
        self.sample_limit = sample_limit

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y, indices = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        positive_idx = np.where(preds == 1)[0]
        positive_idx = indices[positive_idx]
        intermediate_idx = self.counts.loc[(self.counts["correct"] > self.counts["correct"].min()) & (
                self.counts["correct"] < self.counts["correct"].max())]["sample_idx"].to_numpy()
        noise_idx = np.intersect1d(positive_idx, intermediate_idx)
        noise_idx = self.subset_indices(noise_idx, self.sample_limit)

        sort_idx = indices.argsort()
        sort_idx = sort_idx[np.searchsorted(indices, noise_idx, sorter=sort_idx)]

        new_y[sort_idx] = 0

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("intermediate_positive_samples", IntermediatePositiveSamples)
