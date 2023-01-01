import torch.nn
import wandb

from .creation import modules
from .module import Module


class Standard(Module):
    def __init__(self, model, optimizer_args, lr_scheduler_args, *args, **kwargs):
        super().__init__(model, optimizer_args, lr_scheduler_args)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        metrics = self._get_all_metrics(x, y)

        # Log loss and metric
        self.log('train/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('train/accuracy', metrics["accuracy"], on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val/accuracy', summary='max')

        x, y, _ = batch
        metrics = self._get_all_metrics(x, y)

        # Log loss and metric
        self.log('val/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('val/accuracy', metrics["accuracy"], on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def _get_loss(self, logits, y):
        return self.loss_fn(logits, y)

    def _get_accuracy(self, x, y):
        pred = self.model.predict(x)
        accuracy = (pred == y).float().mean()

        return accuracy

    def _get_all_metrics(self, x, y):
        logits = self.model(x)
        loss = self._get_loss(logits, y)

        accuracy = self._get_accuracy(x, y)

        return {"loss": loss, "accuracy": accuracy}


modules.register_builder("standard", Standard)
