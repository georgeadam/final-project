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
        super().__init__(sample_limit)
        self.counts = counts

    def corrupt_helper(self, preds, y, indices):
        y = copy.deepcopy(y)
        corruption_indices = self.get_corruption_indices(preds, indices)
        y[corruption_indices] = 1

        return y

    def get_relevant_indices(self, preds, sample_indices):
        indices = np.where(preds == 0)[0]
        indices = sample_indices[indices]

        return indices

    def get_difficult_indices(self):
        return self.counts.loc[(self.counts["correct"] > self.counts["correct"].min()) & (
                self.counts["correct"] < self.counts["correct"].max())]["sample_idx"].to_numpy()


label_corruptors.register_builder("intermediate_negative_samples", IntermediateNegativeSamples)
