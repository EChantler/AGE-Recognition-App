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
    
class AgeGenderExpressionNet(nn.Module):
    def __init__(self,
                 age_classes=2,
                 gender_classes=2,
                 expr_classes=2,
                 pretrained=True):
        super().__init__()

        # Backbone
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = mobilenet.last_channel  # typically 1280

        # Heads
        self.age_head = nn.Linear(feat_dim, age_classes)
        self.gender_head = nn.Linear(feat_dim, gender_classes)
        self.expr_head = nn.Linear(feat_dim, expr_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        age_logits = self.age_head(x)
        gender_logits = self.gender_head(x)
        expr_logits = self.expr_head(x)

        return {
            "age": age_logits,
            "gender": gender_logits,
            "expression": expr_logits,
        }