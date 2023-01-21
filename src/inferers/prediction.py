import torch
from pytorch_lightning import LightningModule

from .creation import inferers
from .inferer import Inferer


class Prediction(Inferer):
    def make_predictions(self, model, dataloaders):
        module = Module(model)
        results = self.predict(module, dataloaders=dataloaders)

        probs = [results[i][0] for i in range(len(results))]
        probs = torch.cat(probs)

        preds = [results[i][1] for i in range(len(results))]
        preds = torch.cat(preds)

        y = [results[i][2] for i in range(len(results))]
        y = torch.cat(y)

        indices = [results[i][3] for i in range(len(results))]
        indices = torch.cat(indices)

        return probs.numpy(), preds.numpy(), y.numpy(), indices.numpy()


class Module(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx):
        x, y, indices = batch
        probs = self.model.predict_proba(x)
        preds = self.model.predict(x)

        return probs, preds, y, indices


inferers.register_builder("prediction", Prediction)
inferers.register_builder("standard", Prediction)
