import abc


class LabelUpdaterInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def overwrite_update_labels(self, old_labels, new_labels, indices, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def overwrite_train_labels(self, old_labels, new_labels, indices):
        raise NotImplementedError
