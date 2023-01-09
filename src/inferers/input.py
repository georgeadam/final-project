import torch
from pytorch_lightning import LightningModule

from .creation import inferers
from .inferer import Inferer


class Input(Inferer):
    def make_predictions(self, dataloaders):
        module = Module()
        results = self.predict(module, dataloaders=dataloaders)

        x = [results[i][0] for i in range(len(results))]
        x = torch.cat(x)

        indices = [results[i][1] for i in range(len(results))]
        indices = torch.cat(indices)

        return x.numpy(), indices.numpy()


class Module(LightningModule):
    def __init__(self):
        super().__init__()

    def predict_step(self, batch, batch_idx):
        x, y, indices = batch

        return x, indices


inferers.register_builder("input", Input)
