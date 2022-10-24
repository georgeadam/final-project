import abc

import torch.nn.functional


class Model(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self._threshold = 0.5
        pass

    @abc.abstractmethod
    def forward(self, *args):
        raise NotImplementedError

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
