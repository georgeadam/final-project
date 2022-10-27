import copy

import numpy as np

from .corruptor import CorruptorInterface
from .creation import corruptors


class ErrorOscillation(CorruptorInterface):
    def __init__(self, corruption_prob):
        self.corruption_prob = corruption_prob

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y = trainer.predict(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        fn_idx = np.where(np.logical_and(y == 1, preds == 0))[0]
        fn_idx = np.random.choice(fn_idx, size=int(self.corruption_prob * len(fn_idx)),
                                  replace=False)
        new_y[fn_idx] = 0

        data_module.overwrite_current_update_labels(new_y, update_num)


corruptors.register_builder("error_oscillation", ErrorOscillation)
