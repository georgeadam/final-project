from .label_updater import LabelUpdaterInterface


class Null(LabelUpdaterInterface):
    def update_labels(self, old_labels, new_labels, update_num):
        pass
