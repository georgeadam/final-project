import copy

import numpy as np

from .label_corruptor import LabelCorruptorInterface
from .creation import label_corruptors


class NegativeBiasNoise(LabelCorruptorInterface):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        positive_idx = np.where(preds == 1)[0]
        noise_idx = np.random.choice(positive_idx, size=int(self.noise_level * len(positive_idx)), replace=False)
        new_y[noise_idx] = 0

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("negative_bias_noise", NegativeBiasNoise)
