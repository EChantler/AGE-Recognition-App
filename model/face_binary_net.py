import torch
import torch.nn as nn
import torchvision.models as models

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


class MobilenetAgeNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier[-1].in_features
        # Replace classifier head with 3-class output (3 age ranges)
        self.backbone.classifier[-1] = nn.Linear(in_features, 3)

    def forward(self, x):
        return self.backbone(x)


class MobilenetGenderNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier[-1].in_features
        # Replace classifier head with 2-class output (male/female)
        self.backbone.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)
        
class EfficientNetGenderNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier[-1].in_features
        # Replace classifier head with 2-class output (male/female)
        self.backbone.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)


class MobilenetExpressionNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, 7)
        
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
    
class EfficientNetExpressionNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 7)

    def forward(self, x):
        return self.backbone(x)