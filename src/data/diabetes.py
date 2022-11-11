import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from torchvision.transforms import Compose

from src.utils.preprocess import get_numeric_col_indices
from .creation import data_modules
from .data_module import DataModule
from .feeders import feeders
from .splitters import splitters
from .transforms import transforms


class Diabetes(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size)

        self._numeric_cols = None
        self._feeder_args = feeder_args
        self._splitter_args = splitter_args

        self.setup(None)

    def setup(self, stage=None):
        if not self.data_feeder:
            data = fetch_openml("Diabetes-130-Hospitals_(Fairlearn)", as_frame=True, return_X_y=True)
            x, y = data
            x["y"] = y
            x = x.dropna()
            y = x["y"]
            x = x.drop(columns=["y", "readmitted", "readmit_binary"])

            subgroup_cols = ["race", "gender", "age"]
            self._subgroup_features = x[subgroup_cols]

            x = pd.get_dummies(x)
            self._numeric_cols = get_numeric_col_indices(x)

            x = x.to_numpy().astype("float32")
            y = y.to_numpy().astype("float32")
            indices = np.arange(len(x))

            splitter = splitters.create(self._splitter_args.name, **self._splitter_args.params)
            splitted_data = splitter.split_data(x, y, indices)
            self.data_feeder = feeders.create(self._feeder_args.name, **self._feeder_args.params,
                                              splitted_data=splitted_data)
            self._num_updates = self.data_feeder.num_updates
            self._data_dimension = x.shape[1]

            self.update_transforms(0)

    def update_train_transform(self, x):
        scaler = transforms.create("scaler", cols=self._numeric_cols)
        scaler.fit(x)

        tensor = transforms.create("tensor")

        self.train_transform = Compose([scaler, tensor])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        scaler = transforms.create("scaler", cols=self._numeric_cols)
        scaler.fit(x)

        tensor = transforms.create("tensor")

        self.inference_transform = Compose([scaler, tensor])
        self.inference_target_transform = None


data_modules.register_builder("diabetes", Diabetes)
