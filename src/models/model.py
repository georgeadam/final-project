import torch


class Model(torch.nn.Module):
    def __init__(self, warm_start, *args, **kwargs):
        super().__init__()
        self._threshold = 0.5
        self._warm_start = warm_start

    def predict(self, *args):
        probs = self.predict_proba(*args)

        return (probs > self.threshold).int()

    def predict_proba(self, *args):
        with torch.no_grad():
            output = self.forward(*args)

        return torch.sigmoid(output[:, 0])

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold):
        self._threshold = new_threshold

    @property
    def warm_start(self):
        return self._warm_start