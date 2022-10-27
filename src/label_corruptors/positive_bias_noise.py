import copy

import numpy as np

from .label_corruptor import LabelCorruptorInterface
from .creation import label_corruptors


class PositiveBiasNoise(LabelCorruptorInterface):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y = trainer.predict(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        negative_idx = np.where(preds == 0)[0]
        noise_idx = np.random.choice(negative_idx, size=int(self.noise_level * len(negative_idx)), replace=False)
        new_y[noise_idx] = 1

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("positive_bias_noise", PositiveBiasNoise)
