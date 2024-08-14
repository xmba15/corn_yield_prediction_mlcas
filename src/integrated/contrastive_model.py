import pytorch_lightning as pl
from lightly.loss import NegativeCosineSimilarity

from src.models import SimSiam
from src.utils import get_object_from_dict

__all__ = ("SimSiamPl",)


class SimSiamPl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.model = SimSiam(
            self.hparams["model"]["encoder_name"],
            self.hparams["model"]["in_channels"],
            self.hparams["model"]["hidden_size"],
        )
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        z0, p0 = self(x0)
        z1, p1 = self(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.parameters() if x.requires_grad],
        )

        return [
            optimizer,
        ]
