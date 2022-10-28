import copy

import numpy as np

from .label_corruptor import LabelCorruptorInterface
from .creation import label_corruptors


class UniformNoise(LabelCorruptorInterface):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def corrupt(self, module, data_module, trainer, update_num):
        update_batch_dataloader = data_module.current_update_batch_dataloader(update_num)
        _, preds, y = trainer.make_predictions(module, dataloaders=update_batch_dataloader)

        new_y = copy.deepcopy(y)

        noise_idx = np.random.choice(np.arange(len(new_y)), size=int(self.noise_level * len(new_y)), replace=False)
        new_y[noise_idx] = 1 - new_y[noise_idx]

        data_module.overwrite_current_update_labels(new_y, update_num)


label_corruptors.register_builder("uniform_noise", UniformNoise)
