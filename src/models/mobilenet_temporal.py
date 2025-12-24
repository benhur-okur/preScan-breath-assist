# src/models/mobilenet_temporal.py
import torch
import torch.nn as nn
import torchvision.models as tvm


class MobileNetV3Temporal(nn.Module):
    """
    MobileNetV3-Small backbone.
    Frame features -> temporal pooling (mean/max) -> binary logit.
    Input: x [B, T, 3, H, W]
    Output: logits [B]
    """
    def __init__(self, pooling: str = "mean", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.pooling = pooling

        weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = tvm.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 576  # mobilenet_v3_small last channels

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,3,H,W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = self.avgpool(x).flatten(1)  # [B*T, D]

        x = x.view(B, T, -1)  # [B,T,D]

        if self.pooling == "mean":
            x = x.mean(dim=1)             # [B,D]
        elif self.pooling == "max":
            x = x.max(dim=1).values       # [B,D]
        else:
            raise ValueError("pooling must be 'mean' or 'max'")

        logits = self.classifier(x).squeeze(1)  # [B]
        return logits
