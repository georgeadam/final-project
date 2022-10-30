import abc


class ThresholdSelector:
    def __init__(self, desired_rate, desired_value):
        self.desired_rate = desired_rate
        self.desired_value = desired_value
        self.tolerance = 0.01

    # come to think of it, if we want to avoid making predictions on validation data in the model  update loop
    # just for the purpose of getting probs and labels to feed into the threshold selector, then the threshold selector
    # itself will have to take as arguments the module, data_module, and trainer, just like the corruptors do, so that
    # it does a forward pass on the appropriate data
    def select_threshold(self, module, data_module, trainer, update_num):
        val_dataloader = data_module.val_dataloader(update_num)
        probs, _, y, _ = trainer.make_predictions(module, dataloaders=val_dataloader)

        module.model.threshold = self._select_threshold_helper(probs, y)

    @abc.abstractmethod
    def _select_threshold_helper(self, probs, y):
        raise NotImplementedError
