from sklearn.model_selection import train_test_split

from .creation import splitters
from .splitter import SplitterInterface


class Random(SplitterInterface):
    def __init__(self, n_train, n_update, n_test, random_state):
        self.n_train = n_train
        self.n_update = n_update
        self.n_test = n_test
        self.random_state = random_state

    def split_data(self, x, y, indices):
        x_train, x_update, y_train, y_update, indices_train, indices_update = train_test_split(x, y, indices,
                                                                                               train_size=self.n_train,
                                                                                               test_size=self.n_update + self.n_test,
                                                                                               random_state=self.random_state)
        x_update, x_test, y_update, y_test, indices_update, indices_test = train_test_split(x_update, y_update,
                                                                                            indices_update,
                                                                                            train_size=self.n_update,
                                                                                            test_size=self.n_test,
                                                                                            random_state=self.random_state)

        splitted = {"x_train": x_train, "y_train": y_train, "indices_train": indices_train,
                    "x_update": x_update, "y_update": y_update, "indices_update": indices_update,
                    "x_test": x_test, "y_test": y_test, "indices_test": indices_test}

        return splitted


splitters.register_builder("random", Random)
