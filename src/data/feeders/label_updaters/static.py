from .label_updater import LabelUpdaterInterface


class Static(LabelUpdaterInterface):
    def __init__(self, num_updates):
        self._num_updates = num_updates

    def overwrite_update_labels(self, old_labels, new_labels, indices, update_num):
        if len(indices) > 0:
            samples_per_update = int(len(old_labels) / self._num_updates)
            old_labels[(update_num - 1) * samples_per_update: update_num * samples_per_update][indices] = new_labels

    def overwrite_train_labels(self, old_labels, new_labels, indices):
        if len(indices) > 0:
            old_labels[indices] = new_labels