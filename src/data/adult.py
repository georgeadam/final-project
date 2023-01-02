import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from torchvision.transforms import Compose

from src.utils.preprocess import get_numeric_col_indices
from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import transforms


class Adult(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

        self.data_wrapper = EagerDataset
        self._numeric_cols = None

        self.setup(None)

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

    def load_data(self):
        data = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
        x, y = data
        x["y"] = y
        x = x.dropna()
        y = x["y"]
        x = x.drop(columns=["y"])

        subgroup_cols = ["race", "sex", "native-country"]
        self._subgroup_features = x[subgroup_cols]

        x = pd.get_dummies(x)
        self._numeric_cols = get_numeric_col_indices(x)

        x = x.to_numpy().astype("float32")
        y = pd.factorize(y)[0].astype(int)

        return x, y

    def set_stats(self, x, y):
        self._data_dimension = x.shape[1]
        self._num_classes = len(np.unique(y))


data_modules.register_builder("adult", Adult)
