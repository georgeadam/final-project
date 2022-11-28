from sklearn.model_selection import train_test_split

from .creation import splitters
from .splitter import SplitterInterface


class Random(SplitterInterface):
    def __init__(self, n_train, n_update, random_state):
        self.n_train = n_train
        self.n_update = n_update
        self.random_state = random_state

    def split_data(self, x, y, indices):
        n_train, n_update = self._data_sizes_to_counts(self.n_train, self.n_update, len(x))
        x_train, x_update, y_train, y_update, indices_train, indices_update = train_test_split(x,
                                                                                               y,
                                                                                               indices,
                                                                                               train_size=n_train,
                                                                                               random_state=self.random_state)

        if self.n_update == 0:
            x_test, y_test, indices_test = x_update, y_update, indices_update
            x_update, y_update, indices_update = None, None, None
        else:
            x_update, x_test, y_update, y_test, indices_update, indices_test = train_test_split(x_update,
                                                                                                y_update,
                                                                                                indices_update,
                                                                                                train_size=n_update,
                                                                                                random_state=self.random_state)

        splitted = {"x_train": x_train, "y_train": y_train, "indices_train": indices_train,
                    "x_update": x_update, "y_update": y_update, "indices_update": indices_update,
                    "x_test": x_test, "y_test": y_test, "indices_test": indices_test}

        return splitted

    def _data_sizes_to_counts(self, n_train, n_update, total):
        if not isinstance(n_train, int):
            n_train = int(n_train * total)

        if not isinstance(n_update, int):
            n_update = int(n_update * total)

        return n_train, n_update


splitters.register_builder("random", Random)
