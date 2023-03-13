import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10 as TorchvisionCIFAR10

from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import PILImage


class CIFAR10(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

        self.data_wrapper = EagerDataset

        self.setup(None)

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

    def update_corruption_transform(self, x):
        self.corruption_transform = transforms.Compose([transforms.ToTensor()])
        self.corruption_target_transform = None

    def set_stats(self, x, y):
        self._data_dimension = x.shape[1:]
        self._num_classes = len(np.unique(y))

    def _get_x_y(self):
        train_data = TorchvisionCIFAR10(self.data_dir, train=True, download=True)
        test_data = TorchvisionCIFAR10(self.data_dir, train=False, download=True)

        x = np.concatenate([train_data.data, test_data.data])
        y = np.concatenate([np.array(train_data.targets), np.array(test_data.targets)]).astype(int)

        return x, y