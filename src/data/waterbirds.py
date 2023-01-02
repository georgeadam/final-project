import os

import numpy as np
import pandas as pd
from torchvision import transforms

from .creation import data_modules
from .data_module import DataModule
from .feeders import feeders
from .lazy_dataset import LazyDataset
from .splitters import splitters


class Waterbirds(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

        self.data_wrapper = LazyDataset
        self.splits = None
        self.setup(None)

    def update_train_transform(self, x):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                self._data_dimension,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def update_inference_transform(self, x):
        self.inference_transform = transforms.Compose([
            transforms.Resize((self._data_dimension, self._data_dimension)),
            transforms.CenterCrop(self._data_dimension),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_data(self):
        metadata = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        x = [os.path.join(self.data_dir, metadata["img_filename"].values[i]) for i in range(len(metadata))]
        x = np.array(x)
        y = metadata["y"].values
        numeric_splits = metadata["split"].values

        subgroup_cols = ["place"]
        self._subgroup_features = metadata[subgroup_cols]

        self.splits = {}
        self.splits["train"] = np.where(numeric_splits == 0)[0]
        self.splits["update"] = np.where(numeric_splits == 1)[0]
        self.splits["test"] = np.where(numeric_splits == 2)[0]

        y = y.astype(int)

        return x, y

    def create_feeder(self, x, y):
        splitter = splitters.create(self._splitter_args.name, splits=self.splits, **self._splitter_args.params)
        splitted_data = splitter.split_data(x, y)
        feeder = feeders.create(self._feeder_args.name, **self._feeder_args.params, splitted_data=splitted_data)

        return feeder

    def set_stats(self, x, y):
        self._data_dimension = 224
        self._num_classes = len(np.unique(y))


data_modules.register_builder("waterbirds", Waterbirds)
