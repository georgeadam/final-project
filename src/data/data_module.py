import abc
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from settings import ROOT_DIR
from .dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.batch_size = batch_size

        self.data_feeder = None

        self.train_transform = None
        self.train_target_transform = None
        self.inference_transform = None
        self.inference_target_transform = None

        self._num_updates = None
        self._data_dimension = None

    def train_dataloader(self, update_num=None):
        x, y = self.data_feeder.get_train_data(update_num)
        dataset = Dataset(x, y, transform=self.train_transform, target_transform=self.train_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self, update_num=None):
        x, y = self.data_feeder.get_val_data(update_num)
        dataset = Dataset(x, y, transform=self.inference_transform, target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def current_update_batch_dataloader(self, update_num):
        x, y = self.data_feeder.get_current_update_batch(update_num)
        dataset = Dataset(x, y, transform=self.inference_transform, target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def eval_dataloader(self, update_num):
        x, y = self.data_feeder.get_eval_data(update_num)
        dataset = Dataset(x, y, transform=self.inference_transform, target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    @abc.abstractmethod
    def update_train_transform(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def update_inference_transform(self, x):
        raise NotImplementedError

    def update_transforms(self, update_num):
        x, _ = self.data_feeder.get_train_data(update_num)
        self.update_train_transform(x)
        self.update_inference_transform(x)

    @property
    def num_updates(self):
        return self._num_updates

    def overwrite_current_update_labels(self, new_labels, update_num):
        self.data_feeder.overwrite_current_update_labels(new_labels, update_num)

    @property
    def data_dimension(self):
        return self._data_dimension
