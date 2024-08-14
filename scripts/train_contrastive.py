# flake8: noqa: E402
import argparse
import os
import pydoc
import sys

import pytorch_lightning as pl
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import v2

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CombinedDataset, CornYieldDataset, SatelliteImageDataset
from src.models import SimSiam
from src.utils import fix_seed


def get_args():
    parser = argparse.ArgumentParser("train contrastive learning")
    parser.add_argument("--config_path", type=str, default="./config/base_contrastive.yaml")

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    os.makedirs(hparams["output_root_dir"], exist_ok=True)
    fix_seed(hparams["seed"])
    pl.seed_everything(hparams["seed"])

    train_2022_dataset = CornYieldDataset(
        hparams["dataset"]["train_2022"]["root_dir"],
        hparams["dataset"]["train_2022"]["csv_path"],
        hparams["dataset"]["train_2022"]["date_metadata_csv_path"],
    )

    train_2023_dataset = CornYieldDataset(
        hparams["dataset"]["train_2023"]["root_dir"],
        hparams["dataset"]["train_2023"]["csv_path"],
        hparams["dataset"]["train_2023"]["date_metadata_csv_path"],
    )

    val_2023_dataset = CornYieldDataset(
        hparams["dataset"]["validation_2023"]["root_dir"],
        hparams["dataset"]["validation_2023"]["csv_path"],
        hparams["dataset"]["validation_2023"]["date_metadata_csv_path"],
        is_train=False,
    )

    test_2023_dataset = CornYieldDataset(
        hparams["dataset"]["test_2023"]["root_dir"],
        hparams["dataset"]["test_2023"]["csv_path"],
        hparams["dataset"]["test_2023"]["date_metadata_csv_path"],
        is_train=False,
    )

    combined_dataset = CombinedDataset(
        [
            train_2022_dataset,
            train_2023_dataset,
            val_2023_dataset,
            test_2023_dataset,
        ]
    )

    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=hparams["input_size"],
                scale=(0.95, 1.0),
                ratio=(0.95, 1.05),
            ),
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(degrees=180),
        ]
    )

    raw_dataset = SatelliteImageDataset(
        combined_dataset.get_all_image_paths(),
        transforms,
    )

    train_loader = DataLoader(
        raw_dataset,
        batch_size=hparams["train_parameters"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=hparams["num_workers"],
    )

    hparams["optimizer"]["lr"] = hparams["optimizer"]["lr"] * hparams["train_parameters"]["batch_size"] / 256.0

    model = pydoc.locate(hparams["model"]["pl_class"])(hparams)

    trainer = Trainer(
        default_root_dir=hparams["output_root_dir"],
        max_epochs=hparams["trainer"]["max_epochs"],
        log_every_n_steps=hparams["trainer"]["log_every_n_steps"],
        devices=hparams["trainer"]["devices"],
        accelerator=hparams["trainer"]["accelerator"],
        gradient_clip_val=hparams["trainer"]["gradient_clip_val"],
        accumulate_grad_batches=hparams["trainer"]["accumulate_grad_batches"],
        deterministic=True,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(
            save_dir=hparams["output_root_dir"],
            version=f"{hparams['experiment_name']}_"
            f"{hparams['train_parameters']['batch_size']*hparams['trainer']['accumulate_grad_batches']}_"
            f"{hparams['optimizer']['lr']}",
            name=f"{hparams['experiment_name']}",
        ),
        callbacks=[
            ModelCheckpoint(
                save_last=True,
                save_top_k=0,
                every_n_epochs=1,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(
        model,
        train_loader,
    )


if __name__ == "__main__":
    main()
