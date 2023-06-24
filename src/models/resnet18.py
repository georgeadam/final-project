import torch
import torchvision

from .creation import models
from .model import Model


# torchvision.models.resnet.BasicBlock.relu1 = torch.nn.ReLU(inplace=True)
# torchvision.models.resnet.BasicBlock.relu2 = torch.nn.ReLU(inplace=True)
#
#
# def basic_forward(self, x):
#     identity = x
#
#     out = self.conv1(x)
#     out = self.bn1(out)
#     out = self.relu1(out)
#
#     out = self.conv2(out)
#     out = self.bn2(out)
#
#     if self.downsample is not None:
#         identity = self.downsample(x)
#
#     out += identity
#     out = self.relu1(out)
#
#     return out
#
# torchvision.models.resnet.BasicBlock.forward = basic_forward
#
# torchvision.models.resnet.Bottleneck.relu1 = torch.nn.ReLU(inplace=True)
# torchvision.models.resnet.Bottleneck.relu2 = torch.nn.ReLU(inplace=True)


# def bottleneck_forward(self, x):
#     identity = x
#
#     out = self.conv1(x)
#     out = self.bn1(out)
#     out = self.relu(out)
#
#     out = self.conv2(out)
#     out = self.bn2(out)
#     out = self.relu(out)
#
#     out = self.conv3(out)
#     out = self.bn3(out)
#
#     if self.downsample is not None:
#         identity = self.downsample(x)
#
#     out += identity
#     out = self.relu(out)
#
#     return out
#
# torchvision.models.resnet.Bottleneck.forward = bottleneck_forward


class ResNet18(Model):
    def __init__(self, num_classes, warm_start, **kwargs):
        super().__init__(warm_start)
        resnet = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)


        self.num_classes = num_classes

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

    def reset_classification_layer(self):
        resnet = torchvision.models.resnet18(pretrained=False, num_classes=self.num_classes)

        self.fc = resnet.fc


models.register_builder("resnet18", ResNet18)
