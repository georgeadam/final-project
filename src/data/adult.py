import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from torchvision.transforms import Compose

from .creation import data_modules
from .data_module import DataModule
from .feeders import feeders
from .transforms import transforms


class Adult(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args):
        super().__init__(data_dir, batch_size)

        self._numeric_cols = None
        self._feeder_args = feeder_args

        self.setup(None)

    def setup(self, stage=None):
        if not self.data_feeder:
            adult = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

            x, y = adult

            dummy_x = pd.get_dummies(x)
            dummy_cols = dummy_x.select_dtypes(exclude=["float"]).columns

            numeric_cols = dummy_x.columns.difference(dummy_cols)
            numeric_col_indices = []

            for numeric_col in numeric_cols:
                index = dummy_x.columns.get_loc(numeric_col)
                numeric_col_indices.append(index)

            x = dummy_x.to_numpy().astype("float32")
            y = pd.factorize(y)[0].astype(float)
            indices = np.arange(len(x))

            self._numeric_cols = numeric_col_indices
            self.data_feeder = feeders.create(self._feeder_args.name, **self._feeder_args.params,
                                              x=x, y=y, indices=indices)
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


data_modules.register_builder("adult", Adult)
