import os

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import PILImage


class FairFace(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args, task):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

        self.data_wrapper = EagerDataset
        self.task = task

        self.setup(None)

    def update_train_transform(self, x):
        dataset_mean = [0.482, 0.358, 305]
        dataset_std = [0.255, 0.222, 0.217]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.train_transform = transforms.Compose(
            [PILImage(None), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        dataset_mean = [0.482, 0.358, 305]
        dataset_std = [0.255, 0.222, 0.217]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.inference_transform = transforms.Compose([PILImage(None), transforms.ToTensor(), normalize])
        self.inference_target_transform = None

    def set_stats(self, x, y):
        self._data_dimension = x.shape[1:]
        self._num_classes = np.max(y) + 1

    def load_data(self):
        return self._get_x_y()

    def _get_x_y(self):
        with open(os.path.join(self.data_dir, "train_images.npy"), "rb") as f:
            train_images = np.load(f)

        with open(os.path.join(self.data_dir, "val_images.npy"), "rb") as f:
            val_images = np.load(f)

        x = np.concatenate([train_images, val_images])

        train_labels = pd.read_csv(os.path.join(self.data_dir, "fairface_label_train.csv"))
        val_labels = pd.read_csv(os.path.join(self.data_dir, "fairface_label_val.csv"))

        y = pd.concat([train_labels, val_labels])
        self._subgroup_features = y[["age", "gender", "race"]]
        y = y[self.task].to_numpy().astype(int)

        return x, y


data_modules.register_builder("fairface", FairFace)
