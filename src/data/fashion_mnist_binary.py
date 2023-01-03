from .fashion_mnist import FashionMNIST
from .creation import data_modules


class FashionMNISTBinary(FashionMNIST):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

    def load_data(self):
        x, y = self._get_x_y()
        x, y = self._subset_binary(x, y)

        return x, y

    def _subset_binary(self, x, y):
        neg_indices = y == 2
        pos_indices = y == 6

        x = x[neg_indices | pos_indices]

        y[neg_indices] = 0
        y[pos_indices] = 1
        y = y[neg_indices | pos_indices]

        return x, y


data_modules.register_builder("fashion_mnist", FashionMNISTBinary)
data_modules.register_builder("fashion_mnist_binary", FashionMNISTBinary)
