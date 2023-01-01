import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as TorchvisionCIFAR10

from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .feeders import feeders
from .splitters import splitters
from .transforms import PILImage


class CIFAR10(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size)

        self._numeric_cols = None
        self._feeder_args = feeder_args
        self._splitter_args = splitter_args

        self.setup(None)

    def setup(self, stage=None):
        if not self.data_feeder:
            train_data = TorchvisionCIFAR10(self.data_dir, train=True, download=True)
            test_data = TorchvisionCIFAR10(self.data_dir, train=False, download=True)

            x = np.concatenate([train_data.data, test_data.data])
            y = np.concatenate([np.array(train_data.targets), np.array(test_data.targets)])

            neg_indices = y == 1
            pos_indices = y == 9

            x = x[neg_indices | pos_indices]

            y[neg_indices] = 0
            y[pos_indices] = 1
            y = y[neg_indices | pos_indices]
            y = y.astype(int)
            indices = np.arange(len(x))

            splitter = splitters.create(self._splitter_args.name, **self._splitter_args.params)
            splitted_data = splitter.split_data(x, y, indices)
            self.data_feeder = feeders.create(self._feeder_args.name, **self._feeder_args.params,
                                              splitted_data=splitted_data)
            self.data_wrapper = EagerDataset
            self._num_updates = self.data_feeder.num_updates
            self._data_dimension = x.shape[1]
            self._num_classes = 2

            self.update_transforms(0)

    def update_train_transform(self, x):
        dataset_mean = [0.491, 0.482, 0.447]
        dataset_std = [0.247, 0.243, 0.262]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.train_transform = transforms.Compose([PILImage(None),
                                                   transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        dataset_mean = [0.491, 0.482, 0.447]
        dataset_std = [0.247, 0.243, 0.262]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.inference_transform = transforms.Compose([PILImage(None),
                                                       transforms.ToTensor(),
                                                       normalize])
        self.inference_target_transform = None


data_modules.register_builder("cifar10", CIFAR10)
