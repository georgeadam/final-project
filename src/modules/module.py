import abc

from pytorch_lightning import LightningModule


class Module(LightningModule):
    def __init__(self, model, optimizer_args, lr_scheduler_args):
        super().__init__()
        self.model = model
        self._optimizer_args = optimizer_args
        self._lr_scheduler_args = lr_scheduler_args

    def forward(self, batch):
        x, y, idx = batch

        return self.model(x)

    def configure_optimizers(self):
        optimizer = optimizers.create(self._optimizer_args.name, parameters=self.model.parameters(),
                                      **self._optimizer_args.params)
        lr_scheduler = lr_schedulers.create(self._lr_scheduler_args.name, optimizer=optimizer,
                                            **self._lr_scheduler_args.params)

        return [optimizer], [lr_scheduler]

    @abc.abstractmethod
    def _get_loss(self, *args, **kwargs):
        raise NotImplementedError
