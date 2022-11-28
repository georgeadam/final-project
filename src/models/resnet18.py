import torch
import torchvision

from .creation import models
from .model import Model


class ResNet18(Model):
    def __init__(self, warm_start, **kwargs):
        super().__init__(warm_start)
        resnet = torchvision.models.resnet18(pretrained=False, num_classes=1)

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
        x = self.fc(x)

        return x


models.register_builder("resnet18", ResNet18)
