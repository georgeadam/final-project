import abc

import numpy as np


class LabelCorruptor:
    def __init__(self, sample_limit=float("inf"), seed=0):
        self.sample_limit = sample_limit
        self.seed = seed

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        probs, preds, y, sample_indices = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = self.corrupt_helper(probs=probs, preds=preds, y=y, sample_indices=sample_indices)
        data_module.overwrite_current_update_labels(new_y, update_num)

    @abc.abstractmethod
    def corrupt_helper(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_actual_indices(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_potential_indices(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_corruption_indices(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_relevant_indices(self, *args, **kwargs):
        raise NotImplementedError

    def subset_indices(self, indices, sample_limit):
        if sample_limit is not None and len(indices) > sample_limit:
            random_state = np.random.RandomState(self.seed)
            indices = random_state.choice(indices, min(len(indices), sample_limit), replace=False)

        return indices
