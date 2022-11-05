import numpy as np
import pandas as pd
from scipy.io import arff
from torchvision.transforms import Compose

from .creation import data_modules
from .data_module import DataModule
from .feeders import feeders
from .transforms import transforms


class CaliforniaHousing(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args):
        super().__init__(data_dir, batch_size)

        self._numeric_cols = None
        self._feeder_args = feeder_args

        self.setup(None)

    def setup(self, stage=None):
        if not self.data_feeder:
            data = arff.loadarff(self.data_dir)[0]
            data = pd.DataFrame(data)
            data = data.dropna()

            features = [c for c in data.columns if not c.startswith("class")]
            x = data[features]
            y = data["median_house_value"].astype(int)

            dummy_x = pd.get_dummies(x)
            dummy_cols = dummy_x.select_dtypes(exclude=["float"]).columns

            numeric_cols = dummy_x.columns.difference(dummy_cols)
            numeric_col_indices = []

            for numeric_col in numeric_cols:
                index = dummy_x.columns.get_loc(numeric_col)
                numeric_col_indices.append(index)

            x = dummy_x.to_numpy()
            y = y.to_numpy()
            median_price = np.median(y)
            y[y <= median_price] = 0
            y[y > median_price] = 1

            x = x.astype("float32")
            y = y.astype("float32")

            self._numeric_cols = numeric_col_indices
            indices = np.arange(len(x))

            self.data_feeder = feeders.create(self._feeder_args.name, **self._feeder_args.params, x=x, y=y,
                                              indices=indices)
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


data_modules.register_builder("california_housing", CaliforniaHousing)
