import abc
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from settings import ROOT_DIR
from .feeders import feeders
from .splitters import splitters


class DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__()

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.batch_size = batch_size
        self._feeder_args = feeder_args
        self._splitter_args = splitter_args

        self.data_feeder = None
        self.data_wrapper = None

        self.train_transform = None
        self.train_target_transform = None
        self.inference_transform = None
        self.inference_target_transform = None
        self.corruption_transform = None
        self.corruption_target_transform = None

        self._subgroup_features = None
        self._num_updates = feeder_args.params.num_updates
        self._data_dimension = None
        self._num_classes = None

    def setup(self, stage=None):
        if self.data_feeder is None:
            x, y = self.load_data()
            self.data_feeder = self.create_feeder(x, y)
            self.set_stats(x, y)
            self.update_transforms(0)

    def train_dataloader(self, update_num=None):
        x, y, indices = self.data_feeder.get_train_data(update_num)
        dataset = self.data_wrapper(x, y, indices, transform=self.train_transform,
                                    target_transform=self.train_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_inference_dataloader(self, update_num=None):
        x, y, indices = self.data_feeder.get_train_data(update_num)
        dataset = self.data_wrapper(x, y, indices, transform=self.inference_transform,
                                    target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def train_cumulative_inference_dataloader(self, update_num=None):
        x, y, indices = self.data_feeder.get_train_cumulative_data(update_num)
        dataset = self.data_wrapper(x, y, indices, transform=self.inference_transform,
                                    target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def train_initial_dataloader(self):
        x, y, indices = self.data_feeder.get_initial_train_data()
        dataset = self.data_wrapper(x, y, indices, transform=self.inference_transform,
                                    target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self, update_num=None):
        x, y, indices = self.data_feeder.get_val_data(update_num)
        dataset = self.data_wrapper(x, y, indices, transform=self.inference_transform,
                                    target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def current_update_batch_dataloader(self, update_num):
        x, y, indices = self.data_feeder.get_current_update_batch(update_num)
        dataset = self.data_wrapper(x, y, indices, transform=self.inference_transform,
                                    target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def eval_dataloader(self, update_num):
        x, y, indices = self.data_feeder.get_eval_data(update_num)
        dataset = self.data_wrapper(x, y, indices, transform=self.inference_transform,
                                    target_transform=self.inference_target_transform)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_dataloader_by_partition(self, partition, update_num):
        if partition == "train":
            return self.train_dataloader(update_num)
        elif partition == "train_inference":
            return self.train_inference_dataloader(update_num)
        elif partition == "train_cumulative_inference":
            return self.train_cumulative_inference_dataloader(update_num)
        elif partition == "train_initial":
            return self.train_initial_dataloader()
        elif partition == "val":
            return self.val_dataloader(update_num)
        elif partition == "eval":
            return self.eval_dataloader(update_num)
        elif "update" in partition:
            return self.current_update_batch_dataloader(update_num)
        else:
            raise Exception("Incorrect partition {} specified.".format(partition))

    @abc.abstractmethod
    def update_train_transform(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def update_inference_transform(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def update_corruption_transform(self, x):
        raise NotImplementedError

    def update_transforms(self, update_num):
        x, _, _ = self.data_feeder.get_train_data(update_num)
        self.update_train_transform(x)
        self.update_inference_transform(x)
        self.update_corruption_transform(x)

    @property
    def num_updates(self):
        return self._num_updates

    def overwrite_current_update_labels(self, new_labels, indices, update_num):
        self.data_feeder.overwrite_current_update_labels(new_labels, indices, update_num)

    def overwrite_train_labels(self, new_labels, indices):
        self.data_feeder.overwrite_train_labels(new_labels, indices)

    def overwrite_current_update_inputs(self, new_inputs, indices, update_num):
        self.data_feeder.overwrite_current_update_inputs(new_inputs, indices, update_num)

    def overwrite_train_inputs(self, new_inputs, indices):
        self.data_feeder.overwrite_train_inputs(new_inputs, indices)

    @property
    def data_dimension(self):
        return self._data_dimension

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def subgroup_features(self):
        return self._subgroup_features

    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError

    def create_feeder(self, x, y):
        splitter = splitters.create(self._splitter_args.name, **self._splitter_args.params)
        splitted_data = splitter.split_data(x, y)
        feeder = feeders.create(self._feeder_args.name, **self._feeder_args.params, splitted_data=splitted_data)

        return feeder

    @abc.abstractmethod
    def set_stats(self, x, y):
        raise NotImplementedError
