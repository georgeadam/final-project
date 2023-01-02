from .creation import data_modules
from .svhn import SVHN


class SVHNBinary(SVHN):
    def __init__(self, data_dir, batch_size, feeder_args, splitter_args):
        super().__init__(data_dir, batch_size, feeder_args, splitter_args)

    def load_data(self):
        x, y = self._get_x_y()
        x, y = self._subset_binary(x, y)

        return x, y

    def _subset_binary(self, x, y):
        neg_indices = y == 4
        pos_indices = y == 9

        x = x[neg_indices | pos_indices]

        y[neg_indices] = 0
        y[pos_indices] = 1
        y = y[neg_indices | pos_indices]

        return x, y


data_modules.register_builder("svhn", SVHNBinary)
data_modules.register_builder("svhn_binary", SVHNBinary)
