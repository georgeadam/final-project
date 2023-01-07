from .label_updater import LabelUpdaterInterface


class Null(LabelUpdaterInterface):
    def overwrite_update_labels(self, old_labels, new_labels, update_num):
        pass

    def overwrite_train_labels(self, old_labels, new_labels):
        pass
