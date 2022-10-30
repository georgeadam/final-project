import abc

import numpy as np


class LabelCorruptor:
    @abc.abstractmethod
    def corrupt(self, module, data_module, trainer, update_num):
        raise NotImplementedError

    def subset_indices(self, indices, sample_limit):
        if sample_limit is not None and len(indices) > sample_limit:
            indices = np.random_choice(indices, sample_limit, replace=False)

        return indices
