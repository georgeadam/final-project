import numpy as np
import pandas as pd
from scipy.io import arff
from torchvision.transforms import Compose

from src.utils.preprocess import get_numeric_col_indices
from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import transforms


class CaliforniaHousing(DataModule):
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
        data = arff.loadarff(self.data_dir)[0]
        data = pd.DataFrame(data)
        data = data.dropna()

        features = [c for c in data.columns if not c.startswith("class")]
        x = data[features]
        y = data["median_house_value"].astype(int)

        x = pd.get_dummies(x)
        self._numeric_cols = get_numeric_col_indices(x)

        x = x.to_numpy()
        y = y.to_numpy()

        median_price = np.median(y)
        y[y <= median_price] = 0
        y[y > median_price] = 1

        x = x.astype("float32")
        y = y.astype(int)

        return x, y

    def set_stats(self, x, y):
        self._data_dimension = x.shape[1]
        self._num_classes = len(np.unique(y))


data_modules.register_builder("california_housing", CaliforniaHousing)
