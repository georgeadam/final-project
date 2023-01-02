from .fashion_mnist import FashionMNIST
from .creation import data_modules


class FashionMNISTMulticlass(FashionMNIST):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

    def load_data(self):
        return self._get_x_y()


data_modules.register_builder("fashion_mnist_multiclass", FashionMNISTMulticlass)
