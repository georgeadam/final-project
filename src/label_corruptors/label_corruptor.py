import abc

import numpy as np


class LabelCorruptor:
    @abc.abstractmethod
    def corrupt(self, module, data_module, trainer, update_num):
        raise NotImplementedError

    def subset_indices(self, indices, sample_limit):
        if sample_limit is not None and len(indices) > sample_limit:
            random_state = np.random.RandomState(0)
            indices = random_state.choice(indices, min(len(indices), sample_limit), replace=False)

        return indices
