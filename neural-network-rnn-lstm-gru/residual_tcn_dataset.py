import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def get_snapshot_name(sequence_file):
    if "2025-11-15" in sequence_file:
        return "train_nov_predict_from_2025-11-15"
    if "2025-12-15" in sequence_file:
        return "train_dec_predict_from_2025-12-15"
    if "2026-01-14" in sequence_file:
        return "train_jan_predict_from_2026-01-14"
    if "2026-02-13" in sequence_file:
        return "competition_test"

    raise ValueError(f"Не знаю snapshot_name для файла: {sequence_file}")


def get_feature_columns(sequence_file):
    import pyarrow.parquet as pq

    columns = pq.ParquetFile(sequence_file).schema_arrow.names
    technical_columns = {
        "user_id",
        "event_date",
        "day_index",
        "target",
        "cnn_target_residual",
        "cnn_target_residual_clipped",
    }

    feature_columns = []
    for column in columns:
        if column not in technical_columns:
            feature_columns.append(column)

    return feature_columns


def read_residual_for_sequence(
        sequence_file,
        residual_file,
        target_column,
):
    snapshot_name = get_snapshot_name(sequence_file)

    residual = pd.read_csv(
        residual_file,
        usecols=[
            "user_id",
            "snapshot_name",
            "target",
            "target_log",
            "base_hurdle_log",
            target_column,
        ],
    )

    residual = residual.loc[
        residual["snapshot_name"].eq(snapshot_name)
    ].copy()

    if len(residual) == 0:
        raise ValueError(f"Нет residual target для {snapshot_name}")

    return residual


def read_sequence_with_residual(
        sequence_file,
        residual_file,
        feature_columns,
        target_column="cnn_target_residual_clipped",
):
    columns_to_read = ["user_id", "day_index"] + feature_columns
    sequence = pd.read_parquet(sequence_file, columns=columns_to_read)

    residual = read_residual_for_sequence(
        sequence_file,
        residual_file,
        target_column,
    )

    sequence = sequence.merge(
        residual[
            [
                "user_id",
                "target",
                "target_log",
                "base_hurdle_log",
                target_column,
            ]
        ],
        on="user_id",
        how="inner",
    )

    del residual
    gc.collect()

    return sequence


def read_sequence_for_prediction(sequence_file, feature_columns):
    columns_to_read = ["user_id", "day_index"] + feature_columns
    return pd.read_parquet(sequence_file, columns=columns_to_read)


class SequenceResidualDataset(Dataset):
    def __init__(
            self,
            data,
            feature_columns,
            target_column=None,
    ):
        if len(data) == 0:
            raise ValueError("Пустой датасет")

        data = data.sort_values(
            ["user_id", "day_index"],
            kind="stable",
        ).reset_index(drop=True)

        duplicate_mask = (
            data.loc[data["day_index"].ge(0)]
            .duplicated(["user_id", "day_index"])
        )
        if duplicate_mask.any():
            raise ValueError("Есть дубли user_id + day_index")

        self.feature_columns = feature_columns
        self.target_column = target_column
        self.has_target = target_column is not None

        user_values = data["user_id"].to_numpy()
        self.day_indices = data["day_index"].to_numpy(dtype=np.int64)
        self.feature_values = data[feature_columns].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        if self.has_target:
            self.targets = data[target_column].to_numpy(dtype=np.float32)
        else:
            self.targets = np.zeros(len(data), dtype=np.float32)

        user_changes = np.flatnonzero(
            user_values[1:] != user_values[:-1]
        ) + 1

        self.user_starts = np.concatenate(([0], user_changes))
        self.user_ends = np.concatenate((user_changes, [len(data)]))
        self.user_ids = user_values[self.user_starts]

        if len(np.unique(self.user_ids)) != len(self.user_ids):
            raise ValueError("Строки одного user_id должны идти подряд")

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        start = self.user_starts[index]
        end = self.user_ends[index]

        features = np.zeros(
            (180, len(self.feature_columns)),
            dtype=np.float32,
        )

        day_indices = self.day_indices[start:end]
        feature_values = self.feature_values[start:end]

        real_days = (day_indices >= 0) & (day_indices < 180)
        features[day_indices[real_days]] = feature_values[real_days]

        target = self.targets[start]
        user_id = self.user_ids[index]

        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(user_id, dtype=torch.long),
        )


def create_residual_dataloader(
        data,
        feature_columns,
        target_column=None,
        batch_size=128,
        shuffle=True,
):
    dataset = SequenceResidualDataset(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

    return loader
