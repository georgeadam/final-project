import abc

import torch


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._threshold = 0.5

    def predict(self, *args):
        probs = self.predict_proba(*args)

        return (probs[:, 1] > self.threshold).astype(int)

    def predict_proba(self, *args):
        with torch.no_grad():
            output = self.forward(*args)

        return torch.nn.functional.softmax(output, dim=1)

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold):
        self._threshold = new_threshold
