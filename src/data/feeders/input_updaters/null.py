from .input_updater import InputUpdaterInterface


class Null(InputUpdaterInterface):
    def overwrite_update_inputs(self, old_inputs, new_inputs, indices, update_num):
        pass

    def overwrite_train_inputs(self, old_inputs, new_inputs, indices):
        pass
