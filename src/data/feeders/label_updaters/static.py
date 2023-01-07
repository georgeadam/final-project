from .label_updater import LabelUpdaterInterface


class Static(LabelUpdaterInterface):
    def __init__(self, num_updates):
        self._num_updates = num_updates

    def overwrite_update_labels(self, old_labels, new_labels, update_num):
        samples_per_update = int(len(old_labels) / self._num_updates)
        old_labels[(update_num - 1) * samples_per_update: update_num * samples_per_update] = new_labels

    def overwrite_train_labels(self, old_labels, new_labels):
        old_labels[:] = new_labels