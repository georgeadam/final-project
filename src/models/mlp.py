import torch

from .creation import models
from .model import Model


class MLP(Model):
    def __init__(self, data_dimension, layers, activation, warm_start):
        super().__init__(warm_start)

        self.activation = getattr(torch.nn, activation)()

        try:
            self.layers = self._create_layers(layers, data_dimension)
        except Exception:
            pass

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers[-1](x)

        return x

    def embedding(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        return x

    def _create_layers(self, layers, data_dimension):
        if layers == 0:
            fc = torch.nn.ModuleList([torch.nn.Linear(data_dimension, 1)])
        elif layers == 1:
            fc = torch.nn.ModuleList([torch.nn.Linear(data_dimension, 10), torch.nn.Linear(10, 1)])
        elif layers == 2:
            fc = torch.nn.ModuleList(
                [torch.nn.Linear(data_dimension, 20), torch.nn.Linear(20, 10), torch.nn.Linear(10, 1)])
        elif layers > 2:
            initial_hidden_units = 50 * (2 ** layers)
            fc = [torch.nn.Linear(data_dimension, initial_hidden_units)]
            prev_hidden_units = initial_hidden_units

            for i in range(1, layers):
                next_hidden_units = int(initial_hidden_units / (2 ** i))
                fc.append(torch.nn.Linear(prev_hidden_units, next_hidden_units))

                prev_hidden_units = next_hidden_units

            fc.append(torch.nn.Linear(prev_hidden_units, 1))
            fc = torch.nn.ModuleList(fc)
        else:
            raise Exception("Number of layers cannot be negative")

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


models.register_builder("mlp", MLP)
