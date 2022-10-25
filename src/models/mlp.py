import torch

from .creation import models
from .model import Model


class MLP(Model):
    def __init__(self, input_dim, layers, activation):
        super().__init__()

        self.activation = getattr(torch.nn, activation)()

        try:
            self.layers = self._create_layers(layers, input_dim)
        except Exception:
            pass

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[-1](x)

        return x

    def _create_layers(self, layers, input_dim):
        if layers == 0:
            fc = torch.nn.ModuleList([torch.nn.Linear(input_dim, 2)])
        elif layers == 1:
            fc = torch.nn.ModuleList([torch.nn.Linear(input_dim, 10),
                                      torch.nn.Linear(10, 2)])
        elif layers == 2:
            fc = torch.nn.ModuleList([torch.nn.Linear(input_dim, 20),
                                      torch.nn.Linear(20, 10),
                                      torch.nn.Linear(10, 2)])
        elif layers > 2:
            initial_hidden_units = 50 * (2 ** layers)
            fc = [torch.nn.Linear(input_dim, initial_hidden_units)]
            prev_hidden_units = initial_hidden_units

            for i in range(1, layers):
                next_hidden_units = int(initial_hidden_units / (2 ** i))
                fc.append(torch.nn.Linear(prev_hidden_units, next_hidden_units))

                prev_hidden_units = next_hidden_units

            fc.append(torch.nn.Linear(prev_hidden_units, 2))
            fc = torch.nn.ModuleList(fc)
        else:
            raise Exception("Number of layers cannot be negative")

        return fc


models.register_builder("mlp", MLP)
