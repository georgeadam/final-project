import numpy as np
import pandas as pd
from scipy.io import arff
from torchvision.transforms import Compose

from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import transforms


class CIFAR10Tabular(DataModule):
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
        cifar10 = arff.loadarff(self.data_dir)[0]
        cifar10 = pd.DataFrame(cifar10)

        features = [c for c in cifar10.columns if not c.startswith("class")]
        x = cifar10[features]
        y = cifar10["class"].astype(int)
        x = x.to_numpy()
        y = y.to_numpy()

        x /= 255.0

        neg_indices = y == 1
        pos_indices = y == 9

        x = x[neg_indices | pos_indices]
        x = x.astype("float32")

        y[neg_indices] = 0
        y[pos_indices] = 1
        y = y[neg_indices | pos_indices]
        y = y.astype(int)

        return x, y

    def set_stats(self, x, y):
        self._data_dimension = x.shape[1]
        self._num_classes = len(np.unique(y))


data_modules.register_builder("cifar10_tabular", CIFAR10Tabular)
