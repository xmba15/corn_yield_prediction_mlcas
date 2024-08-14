import timm
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from torch import nn

from src.utils import get_extractor_in_features

__all__ = ("SimSiam",)


class SimSiam(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        hidden_size: int,
        pretrained: bool = True,
    ):
        super().__init__()
        in_features = get_extractor_in_features(
            encoder_name,
            in_chans=in_channels,
        )

        if hidden_size > 0:
            self.backbone = nn.Sequential(
                timm.create_model(
                    encoder_name,
                    pretrained=pretrained,
                    num_classes=0,
                    in_chans=in_channels,
                ),
                nn.Linear(in_features, hidden_size),
            )
        else:
            self.backbone = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_channels,
            )

            hidden_size = in_features

        self.projection_head = SimSiamProjectionHead(hidden_size, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p
