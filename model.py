import torch
import torch.nn as nn

# ===== CA Block =====

class CABlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // reduction,
                1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        attention = self.pool(x)

        attention = self.fc(attention)

        return x * attention


from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

# ===== Device =====

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = DeepLabV3_ResNet50_Weights.DEFAULT

# ===== CA-DeepLab =====

class CADeepLab(nn.Module):

    def __init__(self):
        super().__init__()

        self.base = deeplabv3_resnet50(
            weights=weights
        )

        self.ca = CABlock(2048)

    def forward(self, x):

        feats = self.base.backbone(x)

        feature_map = feats["out"]

        feature_map = self.ca(
            feature_map
        )

        out = self.base.classifier(
            feature_map
        )

        return {"out": out}

# ===== Model =====

seg_model = CADeepLab().to(DEVICE)

seg_model.eval()

preprocess = weights.transforms()