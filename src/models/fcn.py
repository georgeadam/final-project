import torch

from .creation import models
from .model import Model


class FCN(Model):
    def __init__(self, data_dimension, num_units, activation, warm_start):
        super().__init__(warm_start)

        self.activation = getattr(torch.nn, activation)()
        self.layers = self._create_layers(num_units, data_dimension)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[-1](x)

        return x

    def _create_layers(self, num_units, data_dimension):
        fc = torch.nn.ModuleList([torch.nn.Linear(data_dimension, num_units),
                                  torch.nn.Linear(num_units, 1)])

        return fc


models.register_builder("fcn", FCN)
