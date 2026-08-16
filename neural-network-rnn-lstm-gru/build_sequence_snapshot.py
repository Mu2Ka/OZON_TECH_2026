import gc

import pandas as pd
from pathlib import Path

df = pd.read_parquet(
    "C:/Users/myska/Downloads/train.parquet"
)

df["event_date"] = pd.to_datetime(df["event_date"])
all_users = (
    df[["user_id"]]
    .drop_duplicates()
    .sort_values("user_id")
    .reset_index(drop=True)
)
daily_features = [features for features in df.columns if features not in ['event_date', 'user_id']]

for feature in daily_features:
    df[feature] = df[feature].astype("float32")

cutoffs = pd.to_datetime([
    "2025-08-17",
    "2025-09-16",
    "2025-10-16",
    "2025-11-15",
    "2025-12-15",
    "2026-01-14",
])


def create_sequence(
        data: pd.DataFrame,
        cutoff: pd.Timestamp,
):
    cutoff = pd.Timestamp(cutoff)
    history_start = cutoff - pd.Timedelta(days=179)
    target_start = cutoff + pd.Timedelta(days=1)
    target_end_exclusive = cutoff + pd.Timedelta(days=31)
    history = data.loc[
        data["event_date"].between(
            history_start,
            cutoff,
            inclusive="both",
        )
    ].copy()
    history['day_index'] = (
        history['event_date'] - history_start
    ).dt.days.astype("int16")
    target = data.loc[
        (data["event_date"] >= target_start)
        & (data["event_date"] < target_end_exclusive)
        , ["user_id", "gmv"]
    ].copy()

    return history, target


def create_dataset(data: pd.DataFrame, cutoffs):
    for cutoff in cutoffs:
        history, target = create_sequence(data=data, cutoff=cutoff)
        target_30d = (
            target.groupby("user_id", as_index=False)["gmv"]
            .sum()
            .rename(columns={"gmv": "target"})
        )
        user_targets = all_users.merge(
            target_30d,
            on="user_id",
            how="left",
        )
        user_targets["target"] = (
            user_targets["target"].fillna(0)
        )
        target_by_user = user_targets.set_index("user_id")["target"]
        history["target"] = (
            history["user_id"]
            .map(target_by_user)
            .fillna(0)
            .astype("float32")
        )
        dataset = history
        user_with_history = history['user_id'].unique()
        missing_users = user_targets.loc[~user_targets["user_id"].isin(user_with_history)].copy()
        missing_users["event_date"] = pd.NaT
        missing_users["day_index"] = -1
        for feature in daily_features:
            missing_users[feature] = 0.0
        missing_users[daily_features] = missing_users[daily_features].astype("float32")
        missing_users["day_index"] = missing_users["day_index"].astype("int16")
        missing_users = missing_users.reindex(columns=dataset.columns)
        dataset = pd.concat(
            [dataset, missing_users], ignore_index=True, copy=False
        )
        Path("data_sequence").mkdir(exist_ok=True)
        output_path = (f"data_sequence/"
                       f"sequence_cutoff{cutoff.date()}.parquet")
        dataset.to_parquet(output_path, index=False)
        print(
            cutoff.date(),
            len(dataset),
            dataset["user_id"].nunique(),
        )
        del history, target, target_30d, user_targets
        del target_by_user, missing_users, dataset
        gc.collect()


if __name__ == "__main__":
    create_dataset(df,  [pd.Timestamp("2026-02-13")])
