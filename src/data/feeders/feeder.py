import abc


class FeederInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_train_data(self, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def get_val_data(self, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def get_current_update_batch(self, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def get_eval_data(self, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def overwrite_current_update_labels(self, new_labels, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def overwrite_train_labels(self, new_labels):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_updates(self):
        raise NotImplementedError
