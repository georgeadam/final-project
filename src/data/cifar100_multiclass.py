from .creation import data_modules

from .cifar100 import CIFAR100


class CIFAR100Multiclass(CIFAR100):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

    def load_data(self):
        return self._get_x_y()


data_modules.register_builder("cifar100_multiclass", CIFAR100Multiclass)