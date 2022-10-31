import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class ErrorAmplification(LabelCorruptor):
    def __init__(self, corruption_prob, sample_limit):
        self.corruption_prob = corruption_prob
        self.sample_limit = sample_limit

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y, _ = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        fp_idx = np.where(np.logical_and(new_y == 0, preds == 1))[0]
        fp_idx = np.random.choice(fp_idx, size=int(self.corruption_prob * len(fp_idx)),
                                  replace=False)
        fp_idx = self.subset_indices(fp_idx, self.sample_limit)

        new_y[fp_idx] = 1

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("error_amplification", ErrorAmplification)
