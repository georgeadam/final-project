from sklearn.model_selection import train_test_split

from .feeder import FeederInterface


class Static(FeederInterface):
    def __init__(self, x, y, indices, n_train, n_update, n_test, val_percentage, num_updates, random_state):
        self.x_train = None
        self.y_train = None
        self.x_update = None
        self.y_update = None
        self.x_test = None
        self.y_test = None

        self.indices_train = None
        self.indices_update = None
        self.indices_test = None

        self._val_percentage = val_percentage
        self._num_updates = num_updates
        self._random_state = random_state
        self._split_data(x, y, indices, n_train, n_update, n_test, random_state)

    @property
    def num_updates(self):
        return self._num_updates

    def get_eval_data(self, update_num):
        return self.x_test, self.y_test, self.indices_test

    def get_current_update_batch(self, update_num):
        samples_per_update = int(len(self.x_update) / self.num_updates)
        x_update = self.x_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]
        y_update = self.y_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]
        indices_update = self.indices_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]

        return x_update, y_update, indices_update

    def overwrite_current_update_labels(self, new_labels, update_num):
        samples_per_update = int(len(self.x_update) / self.num_updates)
        self.y_update[(update_num - 1) * samples_per_update: update_num * samples_per_update] = new_labels

    def _split_data(self, x, y, indices, n_train, n_update, n_test, random_state):
        x_train, x_update, y_train, y_update, indices_train, indices_update = train_test_split(x, y, indices,
                                                                                               train_size=n_train,
                                                                                               test_size=n_update + n_test,
                                                                                               random_state=random_state)
        x_update, x_test, y_update, y_test, indices_update, indices_test = train_test_split(x_update, y_update,
                                                                                            indices_update,
                                                                                            train_size=n_update,
                                                                                            test_size=n_test,
                                                                                            random_state=random_state)

        self.x_train = x_train
        self.y_train = y_train
        self.x_update = x_update
        self.y_update = y_update
        self.x_test = x_test
        self.y_test = y_test

        self.indices_train = indices_train
        self.indices_update = indices_update
        self.indices_test = indices_test