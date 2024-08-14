# flake8: noqa: E402
import argparse
import os
import pydoc
import random
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CornYieldDataset
from src.utils import fix_seed


def get_args():
    parser = argparse.ArgumentParser("train cnn lstm")
    parser.add_argument("--config_path", type=str, default="./config/submit_cnn.yaml")
    parser.add_argument("--test_set", type=str, default="test_2023")
    parser.add_argument("--output_csv", type=str, default="./submit.csv")
    parser.add_argument("--output_with_std_csv", type=str, default="./submit_with_std.csv")
    parser.add_argument("--model_type", type=str, default="cnn")

    return parser.parse_args()


def get_transforms(hparams):
    transforms = {
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


@torch.no_grad()
def main():
    args = get_args()
    with open(args.config_path, encoding="utf-8") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    fix_seed(hparams["seed"])
    pl.seed_everything(hparams["seed"])

    transforms_dict = get_transforms(hparams)

    test_dataset = CornYieldDataset(
        hparams["dataset"][args.test_set]["root_dir"],
        hparams["dataset"][args.test_set]["csv_path"],
        hparams["dataset"][args.test_set]["date_metadata_csv_path"],
        is_train=False,
        transforms=transforms_dict["val"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    all_models = []

    for idx, model_config_dict in enumerate(hparams["model"]):
        model_config_path = model_config_dict[args.model_type]["config_path"]
        model_weights_path = model_config_dict[args.model_type]["weights_path"]
        with open(model_config_path, encoding="utf-8") as f:
            model_hparams = yaml.load(f, Loader=yaml.SafeLoader)

        model = pydoc.locate(model_hparams["model"]["pl_class"]).load_from_checkpoint(
            model_weights_path,
            hparams=model_hparams,
            map_location=device,
        )
        model.to(device)
        model.eval()
        all_models.append(model)

    print(f"number of models: {len(all_models)}")

    df = pd.read_csv(hparams["dataset"][args.test_set]["csv_path"])
    yield_column = []
    yield_std_column = []

    for idx, batch in enumerate(tqdm.tqdm(test_loader)):
        all_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days, _ = [e.to(device) for e in batch]
        cur_corn_yield = []

        def _process_one_pattern(cur_all_data):
            for angle in [0, 90, 180, 270]:
                processed_data = v2.Lambda(lambda img: v2.functional.rotate(img, angle))(cur_all_data)

                for _, model in enumerate(all_models):

                    cur_corn_yield.append(
                        model(processed_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days).item()
                    )

        _process_one_pattern(all_data)
        _process_one_pattern(v2.RandomHorizontalFlip(p=1.0)(all_data))
        _process_one_pattern(v2.RandomVerticalFlip(p=1.0)(all_data))

        cur_corn_yield_std = np.std(cur_corn_yield)
        cur_corn_yield = np.mean(cur_corn_yield)
        yield_column.append(cur_corn_yield)
        yield_std_column.append(cur_corn_yield_std)

    df["yieldPerAcre"] = yield_column
    df.to_csv(args.output_csv, index=False)

    df["yieldPerAcre_std"] = yield_std_column
    df.to_csv(args.output_with_std_csv, index=False)


if __name__ == "__main__":
    main()
