import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import SVHN as TorchvisionSVHN

from .creation import data_modules
from .data_module import DataModule
from .eager_dataset import EagerDataset
from .feeders import feeders
from .splitters import splitters
from .transforms import PILImage, Transpose


class SVHN(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size)

        self._numeric_cols = None
        self._feeder_args = feeder_args
        self._splitter_args = splitter_args

        self.setup(None)

    def setup(self, stage=None):
        if not self.data_feeder:
            train_data = TorchvisionSVHN(self.data_dir, split="train", download=True)
            test_data = TorchvisionSVHN(self.data_dir, split="test", download=True)

            x = np.concatenate([train_data.data, test_data.data])
            y = np.concatenate([np.array(train_data.labels), np.array(test_data.labels)])

            neg_indices = y == 4
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
            self._data_dimension = x.shape[1:]
            self._num_classes = 2

            self.update_transforms(0)

    def update_train_transform(self, x):
        dataset_mean = [0.438, 0.444, 0.473]
        dataset_std = [0.198, 0.201, 0.197]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.train_transform = transforms.Compose([Transpose(),
                                                   PILImage(None),
                                                   transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        dataset_mean = [0.438, 0.444, 0.473]
        dataset_std = [0.198, 0.201, 0.197]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.inference_transform = transforms.Compose([Transpose(),
                                                       PILImage(None),
                                                       transforms.ToTensor(),
                                                       normalize])
        self.inference_target_transform = None


data_modules.register_builder("svhn", SVHN)
