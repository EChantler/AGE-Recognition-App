import torch
import torch.nn as nn
import torchvision.models as models

class FaceBinaryNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)


class MobilenetBinaryNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier[-1].in_features
        # Replace classifier head with 2-class output
        self.backbone.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)
