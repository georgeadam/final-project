import abc


class LabelUpdaterInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update_labels(self, old_labels, new_labels, update_num):
        raise NotImplementedError