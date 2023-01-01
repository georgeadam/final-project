import torch


class Model(torch.nn.Module):
    def __init__(self, warm_start, *args, **kwargs):
        super().__init__()
        self._warm_start = warm_start

    def predict(self, *args):
        with torch.no_grad():
            output = self.forward(*args)

        preds = torch.argmax(output, dim=1)

        return preds

    def predict_proba(self, *args):
        with torch.no_grad():
            output = self.forward(*args)

        preds = torch.argmax(output, dim=1)
        probs = torch.nn.Softmax(dim=1)(output)
        sample_indices = torch.arange(len(probs))

        return probs[sample_indices, preds]

    @property
    def warm_start(self):
        return self._warm_start