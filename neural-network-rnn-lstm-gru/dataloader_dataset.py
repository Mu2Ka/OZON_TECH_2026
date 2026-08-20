import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, data):
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

        self.feature_columns = [
            column
            for column in data.columns
            if column not in [
                "user_id",
                "target",
                "day_index",
                "event_date",
            ]
        ]

        user_values = data["user_id"].to_numpy()
        self.day_indices = data["day_index"].to_numpy(dtype=np.int64)
        self.feature_values = data[self.feature_columns].to_numpy(
            dtype=np.float32,
            copy=False,
        )
        self.targets = data["target"].to_numpy(dtype=np.float32)

        user_changes = np.flatnonzero(
            user_values[1:] != user_values[:-1]
        ) + 1

        self.user_starts = np.concatenate(([0], user_changes))
        self.user_ends = np.concatenate((user_changes, [len(data)]))
        self.user_ids = user_values[self.user_starts]

        if len(np.unique(self.user_ids)) != len(self.user_ids):
            raise ValueError(
                "Строки одного user_id должны идти подряд"
            )

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

        return torch.from_numpy(features), torch.tensor(target)


def create_dataloader_dataset(
        data: pd.DataFrame,
        batch_size=128,
        shuffle=True,
):
    loader = DataLoader(
        MyDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return loader
