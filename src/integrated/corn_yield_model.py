import pytorch_lightning as pl
import torch
from torch.nn import MSELoss
from torchmetrics import MeanSquaredError

from src.models import CornYieldCNN, CornYieldCNNLSTM
from src.utils import get_object_from_dict

__all__ = (
    "CornYieldCNNLSTMPl",
    "CornYieldCNNPl",
)


class CornYieldCNNLSTMPl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.model = CornYieldCNNLSTM(
            hparams["model"]["encoder_name"],
            hparams["model"]["in_channels"],
            hparams["model"]["hidden_size"],
            hparams["model"]["output_size"],
            hparams["model"]["dropout_rate"],
        )

        self.model.load_and_freeze_backbone(
            hparams["model"]["backbone_checkpoint_path"],
        )

        self.criterion = MSELoss()
        self.acc = MeanSquaredError(squared=False)

    def forward(self, seg_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days):
        return self.model(
            seg_data,
            valid_data_mask,
            nitrogen_level,
            location_id,
            norm_passing_days,
        )

    def common_step(self, batch, batch_idx):
        seg_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days, targets = batch
        targets = targets[..., None].float()
        preds = self.model(
            seg_data,
            valid_data_mask,
            nitrogen_level,
            location_id,
            norm_passing_days,
        )

        loss = self.criterion(preds, targets)
        acc = self.acc(preds, targets)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.common_step(batch, batch_idx)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_step(batch, batch_idx)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return acc

    def configure_optimizers(self):
        optimizer = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.parameters() if x.requires_grad],
        )

        scheduler = {
            "scheduler": get_object_from_dict(
                self.hparams["scheduler"],
                optimizer=optimizer,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]


class CornYieldCNNPl(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.model = CornYieldCNN(
            hparams["model"]["encoder_name"],
            hparams["model"]["in_channels"],
            hparams["model"]["hidden_size"],
            hparams["model"]["output_size"],
        )

        self.criterion = MSELoss()
        self.acc = MeanSquaredError(squared=False)

    def forward(self, seg_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days):
        return self.model.forward_sequence(
            seg_data,
            valid_data_mask,
            nitrogen_level,
            location_id,
            norm_passing_days,
        )

    def training_step(self, batch, batch_idx):
        seg_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days, targets = batch
        targets = targets[..., None].float()
        preds = self.model.forward_one_sample(
            seg_data,
            valid_data_mask,
            nitrogen_level,
            location_id,
            norm_passing_days,
        )

        loss = self.criterion(preds, targets)
        acc = self.acc(preds, targets)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        seg_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days, targets = batch
        targets = targets[..., None].float()
        preds = self.model.forward_sequence(
            seg_data,
            valid_data_mask,
            nitrogen_level,
            location_id,
            norm_passing_days,
        )

        loss = self.criterion(preds, targets)
        acc = self.acc(preds, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return acc

    def configure_optimizers(self):
        optimizer = get_object_from_dict(
            self.hparams["optimizer"],
            params=[x for x in self.parameters() if x.requires_grad],
        )

        scheduler = {
            "scheduler": get_object_from_dict(
                self.hparams["scheduler"],
                optimizer=optimizer,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]
