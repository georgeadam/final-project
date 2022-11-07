import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


# the thing is that in order for this to work in the setting of multiple updates, we need to keep track of samples indices
# otherwise we won't know how to match the data in the current batch to the full batch data in counts
# I think that it's fine if we just have indices be compatible for the same random seed and data split, otherwise
# things get a bit tricky. I.e. instead of the indices we return matching the indices of the raw data, they
# instead match the split data. Actually nvm, this is not helpful. In particular, if we later want to take a look
# at impossible samples, we need to figure out the samples in the raw data itself, so we might as well return the index
# from raw data.


class IntermediateNegativeSamples(LabelCorruptor):
    def __init__(self, counts, sample_limit):
        self.counts = counts
        self.sample_limit = sample_limit

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y, indices = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        negative_idx = np.where(preds == 0)[0]
        negative_idx = indices[negative_idx]
        intermediate_idx = self.counts.loc[(self.counts["correct"] > self.counts["correct"].min()) & (
                    self.counts["correct"] < self.counts["correct"].max())]["sample_idx"].to_numpy()
        noise_idx = np.intersect1d(negative_idx, intermediate_idx)
        noise_idx = self.subset_indices(noise_idx, self.sample_limit)

        sort_idx = indices.argsort()
        sort_idx = sort_idx[np.searchsorted(indices, noise_idx, sorter=sort_idx)]

        new_y[sort_idx] = 1

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("intermediate_negative_samples", IntermediateNegativeSamples)
