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
