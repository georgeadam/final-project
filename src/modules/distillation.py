import torch.nn
import wandb

from .creation import modules
from .module import Module


class Distillation(Module):
    def __init__(self, model, original_model, optimizer_args, lr_scheduler_args, alpha, *args, **kwargs):
        super().__init__(model, optimizer_args, lr_scheduler_args)

        self.original_model = original_model
        self.alpha = alpha
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()
        self.distillation_loss_fn = DistillationLoss()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        metrics = self._get_all_metrics(x, y)

        # Log loss and metric
        self.log('train/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('train/loss_ce', metrics["ce_loss"], on_step=False, on_epoch=True)
        self.log('train/loss_distillation', metrics["distillation_loss"], on_step=False, on_epoch=True)
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
        self.log('val/loss_ce', metrics["ce_loss"], on_step=False, on_epoch=True)
        self.log('val/loss_distillation', metrics["distillation_loss"], on_step=False, on_epoch=True)
        self.log('val/accuracy', metrics["accuracy"], on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def _get_ce_loss(self, logits, y):
        return self.ce_loss_fn(logits, y)

    def _get_distillation_loss(self, x, logits):
        with torch.no_grad():
            old_logits = self.original_model(x)

        return self.distillation_loss_fn(logits, old_logits)

    def _get_loss(self, x, logits, y):
        ce_loss = self._get_ce_loss(logits, y)
        distillation_loss = self._get_distillation_loss(x, logits)
        loss = ((1 - self.alpha) * ce_loss) + (self.alpha * distillation_loss)

        return loss, ce_loss, distillation_loss

    def _get_accuracy(self, x, y):
        pred = self.model.predict(x)
        accuracy = (pred == y).float().mean()

        return accuracy

    def _get_all_metrics(self, x, y):
        logits = self.model(x)
        loss, ce_loss, distillation_loss = self._get_loss(x, logits, y)
        accuracy = self._get_accuracy(x, y)

        return {"loss": loss, "ce_loss": ce_loss, "distillation_loss": distillation_loss, "accuracy": accuracy}


class DistillationLoss(object):
    def __init__(self):
        pass

    def __call__(self, logits, old_logits):
        probs = torch.nn.Softmax(dim=1)(logits)
        old_probs = torch.nn.Softmax(dim=1)(old_logits)

        loss = old_probs * (- torch.log(probs))
        loss = torch.sum(loss, dim=1)

        if (loss > 0).sum() == 0:
            return torch.zeros_like(loss).sum()
        else:
            return loss[loss > 0].mean()


modules.register_builder("distillation", Distillation)
