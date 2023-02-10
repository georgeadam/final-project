import os

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import PILImage


class UTKFace(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args, task):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

        self.data_wrapper = EagerDataset
        self.task = task

        self.setup(None)

    def update_train_transform(self, x):
        dataset_mean = [0.596, 0.456, 0.391]
        dataset_std = [0.258, 0.230, 0.226]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.train_transform = transforms.Compose(
            [PILImage(None), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        dataset_mean = [0.596, 0.456, 0.391]
        dataset_std = [0.258, 0.230, 0.226]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.inference_transform = transforms.Compose([PILImage(None), transforms.ToTensor(), normalize])
        self.inference_target_transform = None

    def set_stats(self, x, y):
        self._data_dimension = x.shape[1:]
        self._num_classes = np.max(y) + 1

    def load_data(self):
        return self._get_x_y()

    def _get_x_y(self):
        with open(os.path.join(self.data_dir, "images.npy"), "rb") as f:
            x = np.load(f)

        y = pd.read_csv(os.path.join(self.data_dir, "labels.csv"))
        y = y[self.task].to_numpy().astype(int)

        return x, y


data_modules.register_builder("utkface", UTKFace)
