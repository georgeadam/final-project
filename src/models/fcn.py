import torch

from .creation import models
from .model import Model


class FCN(Model):
    def __init__(self, data_dimension, num_units, num_classes, activation, warm_start):
        super().__init__(warm_start)

        self.activation = getattr(torch.nn, activation)()
        self.layers = self._create_layers(data_dimension, num_units, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers[-1](x)

        return x

    def embedding(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        return x

    def _create_layers(self, data_dimension, num_units, num_classes):
        fc = torch.nn.ModuleList([torch.nn.Linear(data_dimension, num_units), torch.nn.Linear(num_units, num_classes)])

        return fc

    def freeze_classification_layer(self):
        self.layers[-1].requires_grad_(False)

    def freeze_embedding_layers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].requires_grad_(False)

    def unfreeze_classification_layer(self):
        self.layers[-1].requires_grad_(True)

    def unfreeze_embedding_layers(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].requires_grad_(True)


models.register_builder("fcn", FCN)
