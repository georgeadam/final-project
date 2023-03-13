import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import SVHN as TorchvisionSVHN

from .data_module import DataModule
from .eager_dataset import EagerDataset
from .transforms import PILImage, Transpose


class SVHN(DataModule):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

        self.data_wrapper = EagerDataset
        self.setup(None)

    def update_train_transform(self, x):
        dataset_mean = [0.438, 0.444, 0.473]
        dataset_std = [0.198, 0.201, 0.197]

        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

        self.train_transform = transforms.Compose([PILImage(None),
                                                   transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])
        self.train_target_transform = None

    def update_inference_transform(self, x):
        dataset_mean = [0.438, 0.444, 0.473]
        dataset_std = [0.198, 0.201, 0.197]

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
        train_data = TorchvisionSVHN(self.data_dir, split="train", download=True)
        test_data = TorchvisionSVHN(self.data_dir, split="test", download=True)

        x = np.concatenate([train_data.data, test_data.data])
        x = np.transpose(x, (0, 2, 3, 1))
        y = np.concatenate([np.array(train_data.labels), np.array(test_data.labels)]).astype(int)

        return x, y
