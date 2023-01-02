import numpy as np

from .creation import splitters
from .splitter import SplitterInterface


class Precomputed(SplitterInterface):
    def __init__(self, splits):
        self.splits = splits

    def split_data(self, x, y):
        splits = self.splits
        indices = np.arange(len(x))

        x_train, y_train, indices_train = x[splits["train"]], y[splits["train"]], indices[splits["train"]]
        x_update, y_update, indices_update = x[splits["update"]], y[splits["update"]], indices[splits["update"]]
        x_test, y_test, indices_test = x[splits["test"]], y[splits["test"]], indices[splits["test"]]

        splitted = {"x_train": x_train, "y_train": y_train, "indices_train": indices_train,
                    "x_update": x_update, "y_update": y_update, "indices_update": indices_update,
                    "x_test": x_test, "y_test": y_test, "indices_test": indices_test}

        return splitted


splitters.register_builder("precomputed", Precomputed)
