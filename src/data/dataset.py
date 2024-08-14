import os
from datetime import datetime
from typing import List

import cv2
import numpy as np
import pandas as pd
import rasterio as rs
import torch
from torch.utils.data import Dataset

__all__ = (
    "CornYieldDataset",
    "CombinedDataset",
    "SatelliteImageDataset",
    "SubDataset",
)


class SatelliteImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transforms=None,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        with rs.open(self.image_paths[idx]) as ds:
            data = ds.read()

        if self.transforms is not None:
            data = data.clip(0, 10000)
            data = data.astype(np.float32) / 10000
            data = torch.from_numpy(data)
            data = self.transforms(data), self.transforms(data)

        return data


class _BaseDataset(Dataset):
    ALL_TIMEPOINTS = [f"TP{idx+1}" for idx in range(6)]
    ALL_NITROGEN_TREATMENTS = ["Low", "Medium", "High"]
    ALL_LOCATIONS = ["Ames", "MOValley", "Lincoln", "Crawfordsville", "Scottsbluff"]

    def __init__(
        self,
        transforms=None,
    ):
        super().__init__()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        image_paths = self.df.iloc[idx]["satellite_images"]

        all_data = []
        image_shape = None
        for image_path in image_paths:
            if image_path is None:
                all_data.append(None)
                continue

            with rs.open(image_path) as ds:
                data = ds.read()
                all_data.append(data)
                if image_shape is None:
                    image_shape = data.shape

        assert image_shape is not None, self.df.iloc[idx]
        all_data = [e if e is not None else np.zeros(image_shape, dtype=np.uint16) for e in all_data]
        all_data = np.stack([e if e.shape == image_shape else self._resize(e, *image_shape[1:]) for e in all_data])

        if self.transforms is not None:
            all_data = all_data.clip(0, 10000)
            all_data = all_data.astype(np.float32) / 10000
            all_data = torch.from_numpy(all_data)
            all_data = all_data.view(6 * 6, *image_shape[1:])
            all_data = self.transforms(all_data)
            _, new_height, new_width = all_data.shape
            all_data = all_data.view(6, 6, new_height, new_width)

        valid_data_mask = np.array([1 if e is not None else 0 for e in image_paths], dtype=np.uint8)
        nitrogen_level = self.df["nitrogen_level"][idx]
        location_id = self.df["location_id"][idx]
        norm_passing_days = self.df["norm_days_from_planting"][idx]
        corn_yield = self.df["yieldPerAcre"][idx]
        norm_passing_days = np.array(norm_passing_days, dtype=np.float32)

        return all_data, valid_data_mask, nitrogen_level, location_id, norm_passing_days, corn_yield

    def get_all_image_paths(self):
        image_paths = []

        for idx in range(self.__len__()):
            cur_image_paths = self.df.iloc[idx]["satellite_images"]
            cur_image_paths = [image_path for image_path in cur_image_paths if image_path is not None]
            image_paths += cur_image_paths

        return image_paths

    def _resize(self, sample, height, width):
        resized_array = np.zeros((sample.shape[0], height, width), dtype=sample.dtype)
        for i in range(sample.shape[0]):
            resized_array[i] = cv2.resize(sample[i], (width, height), interpolation=cv2.INTER_CUBIC)
        return resized_array


class CornYieldDataset(_BaseDataset):
    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        date_metadata_csv_path: str,
        is_train: bool = True,
        transforms=None,
    ):
        super().__init__(transforms)
        self.root_dir = os.path.expanduser(root_dir)
        self.csv_path = os.path.expanduser(csv_path)
        self.date_metadata_csv_path = os.path.expanduser(date_metadata_csv_path)

        assert os.path.isdir(self.root_dir), f"{self.root_dir} is not a valid directory"
        assert os.path.isfile(self.csv_path), f"{self.csv_path} is not a valid file"
        assert os.path.isfile(self.date_metadata_csv_path), f"{self.date_metadata_csv_path} is not a valid file"

        self.is_train = is_train

        self._process_gt()

    def _process_gt(self):
        self.df = pd.read_csv(self.csv_path)
        if self.is_train:
            self.df = self.df.dropna(subset=["yieldPerAcre"])
        self.df["experiment"] = self.df["experiment"].replace(
            "Hyrbrids",
            "Hybrids",
        )
        locations = set(self.df["location"].to_list())

        self.metadata_df = pd.read_excel(self.date_metadata_csv_path)
        self.metadata_df = self.metadata_df[self.metadata_df["Image"] == "Satellite"]
        self.metadata_df["Location"] = self.metadata_df["Location"].replace(
            "Missouri Valley",
            "MOValley",
        )

        self.df["location_id"] = self.df.apply(
            lambda row: self.ALL_LOCATIONS.index(row["location"]),
            axis=1,
        )
        self.df["satellite_images"] = self.df.apply(
            lambda row: self._get_sequence_paths(row),
            axis=1,
        )
        self.df["nitrogen_level"] = self.df.apply(
            lambda row: self.ALL_NITROGEN_TREATMENTS.index(row["nitrogenTreatment"]),
            axis=1,
        )

        self.df["days_from_planting"] = self.df.apply(
            lambda row: self._get_sequence_days_from_planting_dates(row),
            axis=1,
        )

        self.df["norm_days_from_planting"] = self.df.apply(
            lambda row: [e / 365.0 if e is not None else 0 for e in row["days_from_planting"]],
            axis=1,
        )

    def _get_sequence_paths(
        self,
        df_row,
    ):
        location = df_row["location"]
        time_points = self.metadata_df[self.metadata_df["Location"] == location]["time"].to_list()
        time_points = set(time_points)

        all_paths = []
        for time_point in self.ALL_TIMEPOINTS:
            if time_point not in time_points:
                all_paths.append(None)
            else:
                cur_path = (
                    f'{df_row["location"]}-{time_point}-{df_row["experiment"]}_{df_row["range"]}_{df_row["row"]}.TIF'
                )
                if self.is_train:
                    cur_path = os.path.join(self.root_dir, "Satellite", location, time_point, cur_path)
                else:
                    cur_path = os.path.join(self.root_dir, "Satellite", time_point, cur_path)
                assert os.path.isfile(cur_path), f"{cur_path} is not a valid file"
                all_paths.append(cur_path)

        return all_paths

    def _get_sequence_days_from_planting_dates(
        self,
        df_row,
    ):
        location = df_row["location"]
        time_points = self.metadata_df[self.metadata_df["Location"] == location]["time"].to_list()
        time_points = set(time_points)

        cur_metadata_df = self.metadata_df[self.metadata_df["Location"] == location][["time", "Date"]]
        time_point_to_day_dict = dict(
            zip(
                cur_metadata_df["time"].to_list(),
                cur_metadata_df["Date"].to_list(),
            )
        )

        all_days = []
        date_format = "%Y-%m-%d"
        for time_point in self.ALL_TIMEPOINTS:
            if time_point not in time_points:
                all_days.append(None)
            else:
                df_row["plantingDate"] = pd.to_datetime(df_row["plantingDate"], errors="coerce")
                all_days.append((time_point_to_day_dict[time_point].to_pydatetime() - df_row["plantingDate"]).days)

        return all_days


class CombinedDataset(_BaseDataset):
    def __init__(
        self,
        datasets,
        transforms=None,
    ):
        super().__init__(transforms)
        assert len(datasets) > 0

        self.df = pd.DataFrame(columns=datasets[0].df.columns)
        for dataset in datasets:
            self.df = pd.concat([self.df, dataset.df], ignore_index=True)


class SubDataset(_BaseDataset):
    def __init__(
        self,
        dataset,
        indices,
        transforms=None,
    ):
        super().__init__(transforms)
        self.df = dataset.df.iloc[indices]
