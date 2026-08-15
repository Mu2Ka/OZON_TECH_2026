import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader_dataset(data: pd.DataFrame):


    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data
            self.user_ids = data['user_id'].unique()
            self.groups = data.groupby('user_id', sort=False)
            self.feature_columns = [columns for columns in
                                    self.data.columns not in ['user_id', 'target', 'day_index', '"event_date"']]

        def __len__(self):
            return len(self.user_ids)

        def __getitem__(self, index):
            user_id = self.user_ids[index]
            features = np.zeros(
                (180, len(self.feature_columns)),
                dtype=np.float32,
            )
            user_data = self.groups.get_group(user_id)
            user_data = user_data.sort_values(by=['day_index'], ascending=True)
            real_days = user_data[user_data['day_index'] >= 0]
            day_indices = real_days['day_index'].to_numpy(dtype=np.int64)
            feature_values = real_days[self.feature_columns].to_numpy(dtype=np.float32)
            features[day_indices] = feature_values
            target = np.float32(user_data['target'].iloc[0])
            return torch.from_numpy(features), torch.tensor(target)
    loader = DataLoader(
        MyDataset(data),
        batch_size=512,
        shuffle=True,
        num_workers=4,
    )
    return loader
