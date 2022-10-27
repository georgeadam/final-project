import copy

import numpy as np

from .label_corruptor import LabelCorruptorInterface
from .creation import label_corruptors


class ErrorAmplification(LabelCorruptorInterface):
    def __init__(self, corruption_prob):
        self.corruption_prob = corruption_prob

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y = trainer.predict(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        fp_idx = np.where(np.logical_and(new_y == 0, preds == 1))[0]
        fp_idx = np.random.choice(fp_idx, size=int(self.corruption_prob * len(fp_idx)),
                                  replace=False)
        new_y[fp_idx] = 1

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("error_amplification", ErrorAmplification)
