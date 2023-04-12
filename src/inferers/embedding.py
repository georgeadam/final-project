import torch
from pytorch_lightning import LightningModule

from .creation import inferers
from .inferer import Inferer


class Embedding(Inferer):
    def make_predictions(self, model, dataloaders):
        module = Module(model)
        embeddings = self.predict(module, dataloaders=dataloaders)

        embeddings = torch.cat(embeddings)

        return embeddings.numpy()


class Module(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx):
        x, y, indices = batch
        embeddings = self.model.embedding(x)

        return embeddings


inferers.register_builder("embedding", Embedding)
