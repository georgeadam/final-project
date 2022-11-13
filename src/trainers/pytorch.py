from .creation import trainers
from pytorch_lightning import Trainer
import torch


class PyTorchTrainer(Trainer):
    def __init__(self, update_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_num = update_num

    def make_predictions(self, module, dataloaders):
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



trainers.register_builder("pytorch", PyTorchTrainer)

