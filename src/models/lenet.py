import torch.nn as nn

from .creation import models
from .model import Model


class LeNet(Model):
    def __init__(self, data_dimension, num_classes, warm_start):
        super().__init__(warm_start)

        if data_dimension[1] == 28:
            in_features = 384
        else:
            in_features = 600

        if len(data_dimension) == 3:
            num_channels = data_dimension[2]
        else:
            num_channels = 1

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc3(x)

        return x

    def embedding(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x

    def freeze_classification_layer(self):
        self.fc3.requires_grad_(False)

    def freeze_embedding_layers(self):
        self.conv1.requires_grad_(False)
        self.conv2.requires_grad_(False)

        self.fc1.requires_grad_(False)
        self.fc2.requires_grad_(False)

    def unfreeze_classification_layer(self):
        self.fc3.requires_grad_(True)

    def unfreeze_embedding_layers(self):
        self.conv1.requires_grad_(True)
        self.conv2.requires_grad_(True)

        self.fc1.requires_grad_(True)
        self.fc2.requires_grad_(True)

    def reset_classification_layer(self):
        self.fc3 = nn.Linear(84, self.num_classes)


models.register_builder("lenet", LeNet)

