# flake8: noqa: E402
import argparse
import os
import sys

import yaml

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CombinedDataset, CornYieldDataset


def get_args():
    parser = argparse.ArgumentParser("test corn yield dataset")
    parser.add_argument("--config_path", type=str, default="./config/base.yaml")

    return parser.parse_args()


def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

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

    combined_dataset = CombinedDataset(
        [
            train_2022_dataset,
            train_2023_dataset,
            val_2023_dataset,
        ]
    )


if __name__ == "__main__":
    main()
