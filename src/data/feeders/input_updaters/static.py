from .input_updater import InputUpdaterInterface


class Static(InputUpdaterInterface):
    def __init__(self, num_updates):
        self._num_updates = num_updates

    def overwrite_update_inputs(self, old_inputs, new_inputs, indices, update_num):
        samples_per_update = int(len(old_inputs) / self._num_updates)
        old_inputs[(update_num - 1) * samples_per_update: update_num * samples_per_update][indices] = new_inputs

    def overwrite_train_inputs(self, old_inputs, new_inputs, indices):
        old_inputs[indices] = new_inputs