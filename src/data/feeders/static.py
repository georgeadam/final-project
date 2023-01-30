import abc

from sklearn.model_selection import train_test_split

from .feeder import FeederInterface


class Static(FeederInterface):
    def __init__(self, splitted_data, val_percentage, num_updates, random_state):
        self.x_train = splitted_data["x_train"]
        self.y_train = splitted_data["y_train"]
        self.indices_train = splitted_data["indices_train"]

        self.x_update = splitted_data["x_update"]
        self.y_update = splitted_data["y_update"]
        self.indices_update = splitted_data["indices_update"]

        self.x_test = splitted_data["x_test"]
        self.y_test = splitted_data["y_test"]
        self.indices_test = splitted_data["indices_test"]

        self._val_percentage = val_percentage
        self._num_updates = num_updates
        self._random_state = random_state

        self._input_updater = None
        self._label_updater = None
        self._batch_fetcher = None

    @property
    def num_updates(self):
        return self._num_updates

    def get_train_data(self, update_num):
        return self._get_train_data(update_num)

    def get_initial_train_data(self):
        return self.x_train, self.y_train, self.indices_train

    def get_val_data(self, update_num):
        return self._get_val_data(update_num)

    def get_eval_data(self, update_num):
        return self.x_test, self.y_test, self.indices_test

    def get_current_update_batch(self, update_num):
        return self._batch_fetcher.fetch(self.x_update, self.y_update, self.indices_update, update_num)

    def overwrite_current_update_labels(self, new_labels, update_num):
        self._label_updater.overwrite_update_labels(self.y_update, new_labels, update_num)

    def overwrite_train_labels(self, new_labels):
        self._label_updater.overwrite_train_labels(self.y_train, new_labels)

    def overwrite_current_update_inputs(self, new_inputs, indices, update_num):
        self._input_updater.overwrite_update_inputs(self.x_update, new_inputs, indices, update_num)

    def overwrite_train_inputs(self, new_inputs, indices):
        self._input_updater.overwrite_train_inputs(self.x_train, new_inputs, indices)

    @abc.abstractmethod
    def _get_train_data(self, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_val_data(self, update_num):
        raise NotImplementedError