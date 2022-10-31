import copy

import numpy as np

from .creation import label_corruptors
from .label_corruptor import LabelCorruptor


class PositiveBiasNoise(LabelCorruptor):
    def __init__(self, noise_level, sample_limit):
        self.noise_level = noise_level
        self.sample_limit = sample_limit

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y, _ = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        negative_idx = np.where(preds == 0)[0]
        noise_idx = np.random.choice(negative_idx, size=int(self.noise_level * len(negative_idx)), replace=False)
        noise_idx = self.subset_indices(noise_idx, self.sample_limit)
        new_y[noise_idx] = 1

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("positive_bias_noise", PositiveBiasNoise)
