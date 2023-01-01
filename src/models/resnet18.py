import torch
import torchvision

from .creation import models
from .model import Model


class ResNet18(Model):
    def __init__(self, num_classes, warm_start, **kwargs):
        super().__init__(warm_start)
        resnet = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.avgpool = resnet.avgpool

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.fc = resnet.fc

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)

        return x

    def embedding(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def freeze_classification_layer(self):
        self.fc.requires_grad_(False)

    def freeze_embedding_layers(self):
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)

        self.layer1.requires_grad_(False)
        self.layer2.requires_grad_(False)
        self.layer3.requires_grad_(False)
        self.layer4.requires_grad_(False)

    def unfreeze_classification_layer(self):
        self.fc.requires_grad_(True)

    def unfreeze_embedding_layers(self):
        self.conv1.requires_grad_(True)
        self.bn1.requires_grad_(True)

        self.layer1.requires_grad_(True)
        self.layer2.requires_grad_(True)
        self.layer3.requires_grad_(True)
        self.layer4.requires_grad_(True)


models.register_builder("resnet18", ResNet18)
