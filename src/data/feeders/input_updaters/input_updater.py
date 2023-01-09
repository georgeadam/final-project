import abc


class InputUpdaterInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def overwrite_update_inputs(self, old_inputs, new_inputs, indices, update_num):
        raise NotImplementedError

    @abc.abstractmethod
    def overwrite_train_inputs(self, old_inputs, new_inputs, indices):
        raise NotImplementedError
