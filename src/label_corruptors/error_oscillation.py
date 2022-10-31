import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class ErrorOscillation(LabelCorruptor):
    def __init__(self, corruption_prob, sample_limit):
        self.corruption_prob = corruption_prob
        self.sample_limit = sample_limit

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y, _ = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        fn_idx = np.where(np.logical_and(y == 1, preds == 0))[0]
        fn_idx = np.random.choice(fn_idx, size=int(self.corruption_prob * len(fn_idx)),
                                  replace=False)
        fn_idx = self.subset_indices(fn_idx, self.sample_limit)

        new_y[fn_idx] = 0

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("error_oscillation", ErrorOscillation)
