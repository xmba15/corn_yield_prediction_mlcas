# flake8: noqa: E402
import argparse
import os
import pydoc
import random
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tqdm
import yaml
from lightgbm import LGBMRegressor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBRegressor

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, "../"))
from src.data import CombinedDataset, CornYieldDataset, SubDataset
from src.utils import fix_seed


def get_args():
    parser = argparse.ArgumentParser("train cnn lstm")
    parser.add_argument("--config_path", type=str, default="./config/base_ml.yaml")

    return parser.parse_args()


def setup_train_val_split(
    corn_yields,
    hparams,
):
    num_splits = hparams["dataset"]["n_splits"]
    kf = StratifiedKFold(
        n_splits=num_splits,
        shuffle=True,
        random_state=hparams["seed"],
    )

    num_bins = int(np.floor(len(corn_yields) / num_splits))
    bins = pd.cut(corn_yields, bins=num_bins, labels=False)

    return list(
        kf.split(
            range(len(corn_yields)),
            bins,
        )
    )


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

    train_dataset = CombinedDataset(
        [
            train_2022_dataset,
            train_2023_dataset,
        ]
    )

    test_dataset = CornYieldDataset(
        hparams["dataset"]["test_2023"]["root_dir"],
        hparams["dataset"]["test_2023"]["csv_path"],
        hparams["dataset"]["test_2023"]["date_metadata_csv_path"],
        is_train=False,
    )

    if os.path.isfile(hparams["dataset"]["feature_data_path"]) and os.path.isfile(
        hparams["dataset"]["yield_data_path"]
    ):
        with open(hparams["dataset"]["feature_data_path"], "rb") as _file:
            all_features = np.load(_file)

        with open(hparams["dataset"]["yield_data_path"], "rb") as _file:
            corn_yields = np.load(_file)
    else:
        all_features = []
        corn_yields = []
        for idx in tqdm.tqdm(range(len(train_dataset))):
            features, corn_yield = train_dataset.get_engineered_feature(idx)
            all_features.append(features)
            corn_yields.append(corn_yield)

        all_features = np.array(all_features)
        corn_yields = np.array(corn_yields)

        with open(hparams["dataset"]["feature_data_path"], "wb") as _file:
            np.save(_file, all_features)

        with open(hparams["dataset"]["yield_data_path"], "wb") as _file:
            np.save(_file, corn_yields)

    if os.path.isfile(hparams["dataset"]["test_feature_data_path"]):
        with open(hparams["dataset"]["test_feature_data_path"], "rb") as _file:
            test_features = np.load(_file)
    else:
        test_features = []
        for idx in tqdm.tqdm(range(len(test_dataset))):
            features, _ = test_dataset.get_engineered_feature(idx)
            test_features.append(features)

        test_features = np.array(test_features)
        with open(hparams["dataset"]["test_feature_data_path"], "wb") as _file:
            np.save(_file, test_features)

    corn_yields = np.log(corn_yields)

    # Initial parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "min_child_samples": [1, 5, 10],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    lgbm = LGBMRegressor(random_state=42)

    grid_search = GridSearchCV(
        lgbm,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(
        all_features,
        corn_yields,
    )

    best_params = grid_search.best_params_

    # Print the best parameters and score
    print("Best parameters:", best_params)


if __name__ == "__main__":
    main()
