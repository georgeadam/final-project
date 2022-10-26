import torch
from sklearn.model_selection import train_test_split

from .creation import feeders
from .feeder import FeederInterface


class Static(FeederInterface):
    def __init__(self, x, y, n_train, n_update, n_test, val_percentage, num_updates, random_state):
        self.x_train = None
        self.y_train = None
        self.x_update = None
        self.y_update = None
        self.x_test = None
        self.y_test = None

        self._val_percentage = val_percentage
        self._num_updates = num_updates
        self._random_state = random_state
        self._split_data(x, y, n_train, n_update, n_test, random_state)

    @property
    def num_updates(self):
        return self._num_updates

    def get_train_data(self, update_num):
        x, y = self._get_all_cumulative_data(update_num)
        x, _, y, _ = train_test_split(x, y, test_size=self._val_percentage, random_state=self._random_state)

        return x, y

    def get_val_data(self, update_num):
        x, y = self._get_all_cumulative_data(update_num)
        _, x, _, y = train_test_split(x, y, test_size=self._val_percentage, random_state=self._random_state)

        return x, y

    def get_eval_data(self, update_num):
        return self.x_test, self.y_test

    def get_current_update_batch(self, update_num):
        samples_per_update = int(len(self.x_update) / self.num_updates)
        x_update = self.x_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]
        y_update = self.y_update[(update_num - 1) * samples_per_update: update_num * samples_per_update]

        return x_update, y_update

    def overwrite_current_update_labels(self, update_num, new_labels):
        samples_per_update = int(len(self.x_update) / self.num_updates)
        self.y_update[(update_num - 1) * samples_per_update: update_num * samples_per_update] = new_labels

    def _get_all_cumulative_data(self, update_num):
        x_train, y_train = self.x_train, self.y_train

        if update_num == 0:
            return x_train, y_train

        samples_per_update = int(len(self.x_update) / self.num_updates)

        x_update = self.x_update[: update_num * samples_per_update]
        y_update = self.y_update[: update_num * samples_per_update]

        x = torch.cat([x_train, x_update])
        y = torch.cat([y_train, y_update])

        return x, y

    def _split_data(self, x, y, n_train, n_update, n_test, random_state):
        x_train, x_update, y_train, y_update = train_test_split(x, y, train_size=n_train, test_size=n_update + n_test,
                                                                random_state=random_state)
        x_update, x_test, y_update, y_test = train_test_split(x_update, y_update, train_size=n_update, test_size=n_test,
                                                              random_state=random_state)

        self.x_train = x_train
        self.y_train = y_train
        self.x_update = x_update
        self.y_update = y_update
        self.x_test = x_test
        self.y_test = y_test


feeders.register_builder("static", Static)
