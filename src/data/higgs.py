import numpy as np
import pandas as pd
from scipy.io import arff
from torchvision.transforms import Compose

from .creation import data_modules
from .data_module import DataModule
from .feeders import feeders
from .transforms import transforms


class Higgs(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args):
        super().__init__(data_dir, batch_size)

        self._feeder_args = feeder_args

        self.setup(None)

    def setup(self, stage=None):
        if not self.data_feeder:
            higgs = arff.loadarff(self.data_dir)[0]
            higgs = pd.DataFrame(higgs)
            higgs = higgs.dropna()

            features = [c for c in higgs.columns if not c.startswith("class")]
            x = higgs[features]
            y = higgs["class"].astype(int)

            x = x.to_numpy()
            y = y.to_numpy()

            x = x.astype("float32")
            y = y.astype("float32")

            indices = np.arange(len(x))

            self.data_feeder = feeders.create(self._feeder_args.name, **self._feeder_args.params, x=x, y=y,
                                              indices=indices)
            self._num_updates = self.data_feeder.num_updates
            self._data_dimension = x.shape[1]

            self.update_transforms(0)

    def update_train_transform(self, x):
        scaler = transforms.create("scaler", cols=None)
        scaler.fit(x)

        tensor = transforms.create("tensor")

        self.train_transform = Compose([scaler, tensor])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        scaler = transforms.create("scaler", cols=None)
        scaler.fit(x)

        tensor = transforms.create("tensor")

        self.inference_transform = Compose([scaler, tensor])
        self.inference_target_transform = None


data_modules.register_builder("higgs", Higgs)
