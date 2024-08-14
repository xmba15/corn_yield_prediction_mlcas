# flake8: noqa: E402
import argparse
import os
import pydoc
import random
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CombinedDataset, CornYieldDataset, SubDataset
from src.utils import fix_seed


def get_args():
    parser = argparse.ArgumentParser("train cnn lstm")
    parser.add_argument("--config_path", type=str, default="./config/base.yaml")

    return parser.parse_args()


def setup_train_val_split(
    dataset,
    hparams,
):
    num_splits = hparams["dataset"]["n_splits"]
    kf = StratifiedKFold(
        n_splits=num_splits,
        shuffle=True,
        random_state=hparams["seed"],
    )

    corn_yields = dataset.df["yieldPerAcre"].to_list()
    num_bins = int(np.floor(len(corn_yields) / num_splits))
    bins = pd.cut(corn_yields, bins=num_bins, labels=False)

    train_indices, val_indices = list(
        kf.split(
            range(len(dataset)),
            bins,
        )
    )[hparams["dataset"]["fold_th"]]

    return train_indices, val_indices


def get_transforms(hparams):
    transforms = {
        "train": v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=hparams["input_size"],
                    scale=(0.95, 1.0),
                    ratio=(0.95, 1.05),
                ),
                v2.RandomVerticalFlip(),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(degrees=180),
                # v2.Lambda(lambda img: v2.functional.rotate(img, random.choice([0, 90, 180, 270]))),
            ],
        ),
        "val": v2.Compose(
            [
                v2.Resize(
                    size=(
                        hparams["input_size"],
                        hparams["input_size"],
                    )
                ),
            ],
        ),
    }

    return transforms


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

    train_2022_indices, val_2022_indices = setup_train_val_split(
        train_2022_dataset,
        hparams,
    )

    train_2023_indices, val_2023_indices = setup_train_val_split(
        train_2023_dataset,
        hparams,
    )

    transforms_dict = get_transforms(hparams)

    train_dataset = CombinedDataset(
        [
            SubDataset(
                train_2022_dataset,
                train_2022_indices,
            ),
            SubDataset(
                train_2023_dataset,
                train_2023_indices,
            ),
        ],
        transforms_dict["train"],
    )

    val_dataset = CombinedDataset(
        [
            SubDataset(
                train_2022_dataset,
                val_2022_indices,
            ),
            SubDataset(
                train_2023_dataset,
                val_2023_indices,
            ),
        ],
        transforms_dict["val"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["train_parameters"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=hparams["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams["val_parameters"]["batch_size"],
        num_workers=hparams["num_workers"],
    )

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
                monitor="val_acc",
                mode="min",
                save_top_k=1,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    if hparams["trainer"]["resume_from_checkpoint"] is not None and os.path.isfile(
        hparams["trainer"]["resume_from_checkpoint"]
    ):
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=hparams["trainer"]["resume_from_checkpoint"],
        )
    else:
        trainer.fit(
            model,
            train_loader,
            val_loader,
        )


if __name__ == "__main__":
    main()
