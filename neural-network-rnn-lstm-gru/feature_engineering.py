import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


HISTORY_DAYS = 180
BATCH_SIZE = 250_000
INPUT_DIR = "data_sequence"
OUTPUT_DIR = "data_sequence_features"

BASE_COLUMNS = [
    "event_date",
    "user_id",
    "day_index",
    "search",
    "cat",
    "has_search_to_cart",
    "has_search_to_ord",
    "has_cat_to_cart",
    "has_cat_to_ord",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "gmv_search",
    "gmv_cat",
    "to_cart",
    "to_ord",
    "gmv",
    "searches",
]
TECH_COLUMNS = {"event_date", "user_id", "day_index", "target"}
VALUE_COLUMNS = [
    column
    for column in BASE_COLUMNS
    if column not in {"event_date", "user_id", "day_index"}
]
MAIN_COLUMNS = ["gmv", "searches", "to_cart", "to_ord"]
RATIO_COLUMNS = ["search_to_cart", "search_to_ord", "cat_to_ord", "gmv_search"]
LAGS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30, 60, 90]
ROLLING_DETAIL_WINDOWS = [7, 30, 90]
EWM_HALF_LIVES = [7, 30]
CUTOFF_STATE_COLUMN = "cutoff_state_row"

FILES = [
    (
        "sequence_cutoff2025-08-17.parquet",
        "sequence_cutoff2025-08-17_features.parquet",
        "2025-08-17",
    ),
    (
        "sequence_cutoff2025-09-16.parquet",
        "sequence_cutoff2025-09-16_features.parquet",
        "2025-09-16",
    ),
    (
        "sequence_cutoff2025-10-16.parquet",
        "sequence_cutoff2025-10-16_features.parquet",
        "2025-10-16",
    ),
    (
        "sequence_cutoff2025-11-15.parquet",
        "sequence_cutoff2025-11-15_features.parquet",
        "2025-11-15",
    ),
    (
        "sequence_cutoff2025-12-15.parquet",
        "sequence_cutoff2025-12-15_features.parquet",
        "2025-12-15",
    ),
    (
        "sequence_cutoff2026-01-14.parquet",
        "sequence_cutoff2026-01-14_features.parquet",
        "2026-01-14",
    ),
    (
        "sequence_cutoff2026-02-13.parquet",
        "sequence_cutoff2026-02-13_features.parquet",
        "2026-02-13",
    ),
]


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype="float32")
    denominator = np.asarray(denominator, dtype="float32")
    result = np.zeros(len(numerator), dtype="float32")

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )

    return result


def bounded_change(recent, previous):
    return safe_divide(
        np.asarray(recent, dtype="float32") - np.asarray(previous, dtype="float32"),
        np.abs(recent) + np.abs(previous) + 1,
    )


def check_columns(df):
    missing = [column for column in BASE_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"Нет обязательных колонок: {missing}")


def add_cutoff_state_rows(df, cutoff):
    df[CUTOFF_STATE_COLUMN] = np.float32(0.0)

    users_with_last_day = df.loc[
        df["day_index"].eq(HISTORY_DAYS - 1),
        "user_id",
    ].drop_duplicates()
    first_rows = df.drop_duplicates("user_id", keep="first")
    missing_last_day = ~first_rows["user_id"].isin(users_with_last_day)

    if not missing_last_day.any():
        return df

    state_rows = first_rows.loc[missing_last_day].copy()
    state_rows["event_date"] = pd.Timestamp(cutoff)
    state_rows["day_index"] = np.int16(HISTORY_DAYS - 1)
    state_rows[CUTOFF_STATE_COLUMN] = np.float32(1.0)
    state_rows[VALUE_COLUMNS] = np.float32(0.0)

    return pd.concat([df, state_rows], ignore_index=True)


def prepare_batch(df, cutoff):
    check_columns(df)

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    if "target" not in df.columns:
        df["target"] = np.float32(0.0)

    df = add_cutoff_state_rows(df, cutoff)
    df = df.sort_values(["user_id", "day_index"], kind="stable")
    df = df.reset_index(drop=True)

    duplicates = (
        df.loc[df["day_index"].ge(0)]
        .duplicated(["user_id", "day_index"])
    )

    if duplicates.any():
        raise ValueError("Есть дубли user_id + day_index")

    return df


def put_feature(features, name, row_mask, values):
    full_column = np.zeros(len(row_mask), dtype="float32")
    full_column[np.asarray(row_mask)] = np.asarray(values, dtype="float32")
    features[name] = full_column


def add_calendar_features(df, features):
    real_mask = df["day_index"].ge(0).to_numpy()
    observed_mask = (
        df["day_index"].ge(0) & df[CUTOFF_STATE_COLUMN].eq(0)
    ).to_numpy()
    real = df.loc[real_mask]

    features["observed_day"] = observed_mask.astype("float32")
    put_feature(
        features,
        "day_index_norm",
        real_mask,
        real["day_index"] / (HISTORY_DAYS - 1),
    )

    gaps = real.groupby("user_id", sort=False)["day_index"].diff()
    first_gap = gaps.isna()
    gaps.loc[first_gap] = real.loc[first_gap, "day_index"] + 1
    put_feature(features, "days_since_prev_observed", real_mask, gaps)

    day_of_week = real["event_date"].dt.dayofweek
    day_of_year = real["event_date"].dt.dayofyear

    put_feature(
        features,
        "calendar_is_weekend",
        real_mask,
        day_of_week.ge(5).astype("float32"),
    )
    put_feature(
        features,
        "calendar_day_of_week_sin",
        real_mask,
        np.sin(2 * np.pi * day_of_week / 7),
    )
    put_feature(
        features,
        "calendar_day_of_week_cos",
        real_mask,
        np.cos(2 * np.pi * day_of_week / 7),
    )
    put_feature(
        features,
        "calendar_day_of_year_sin",
        real_mask,
        np.sin(2 * np.pi * day_of_year / 365.25),
    )
    put_feature(
        features,
        "calendar_day_of_year_cos",
        real_mask,
        np.cos(2 * np.pi * day_of_year / 365.25),
    )


def add_lag_features(df, features):
    real_mask = df["day_index"].ge(0).to_numpy()
    observed_mask = (
        df["day_index"].ge(0) & df[CUTOFF_STATE_COLUMN].eq(0)
    ).to_numpy()
    real = df.loc[real_mask]
    observed = df.loc[observed_mask]

    real_index = pd.MultiIndex.from_frame(real[["user_id", "day_index"]])
    observed_index = pd.MultiIndex.from_frame(observed[["user_id", "day_index"]])
    observed_lookup = pd.Series(1.0, index=observed_index)

    value_lookup = {}
    for column in MAIN_COLUMNS:
        value_lookup[column] = pd.Series(real[column].to_numpy(), index=real_index)

    for lag in LAGS:
        lag_index = pd.MultiIndex.from_arrays(
            [real["user_id"], real["day_index"] - lag]
        )

        put_feature(
            features,
            f"lag_observed_{lag}d",
            real_mask,
            observed_lookup.reindex(lag_index).fillna(0).to_numpy(),
        )

        for column in MAIN_COLUMNS:
            put_feature(
                features,
                f"lag_{column}_{lag}d",
                real_mask,
                value_lookup[column].reindex(lag_index).fillna(0).to_numpy(),
            )


def rolling_table(real, columns, window):
    table = real[["user_id", "event_date"]].copy()

    for column in columns:
        values = real[column].astype("float32")
        table[f"{column}_sum"] = values
        table[f"{column}_square"] = values * values
        table[f"{column}_nonzero"] = values.gt(0).astype("float32")

    rolling_columns = [
        column
        for column in table.columns
        if column not in {"user_id", "event_date"}
    ]

    rolled = (
        table.set_index("event_date")
        .groupby("user_id", sort=False)[rolling_columns]
        .rolling(f"{window}D", closed="right")
        .sum()
        .reset_index()
    )

    return real[["user_id", "event_date"]].merge(
        rolled,
        on=["user_id", "event_date"],
        how="left",
        sort=False,
        validate="one_to_one",
    )


def add_rolling_features(df, features):
    real_mask = df["day_index"].ge(0).to_numpy()
    real = df.loc[real_mask].copy()
    rolling_sums = {}

    for window in ROLLING_WINDOWS:
        rolled = rolling_table(real, MAIN_COLUMNS, window)
        exposure = np.minimum(window, real["day_index"] + 1).clip(lower=1)

        for column in MAIN_COLUMNS:
            rolling_sum = rolled[f"{column}_sum"].fillna(0).to_numpy()
            rolling_sums[(column, window)] = rolling_sum

            put_feature(
                features,
                f"roll_{column}_sum_{window}d",
                real_mask,
                rolling_sum,
            )

            if window not in ROLLING_DETAIL_WINDOWS:
                continue

            square_sum = rolled[f"{column}_square"].fillna(0).to_numpy()
            nonzero_sum = rolled[f"{column}_nonzero"].fillna(0).to_numpy()
            mean = rolling_sum / exposure
            variance = square_sum / exposure - mean * mean

            put_feature(
                features,
                f"roll_{column}_std_{window}d",
                real_mask,
                np.sqrt(np.maximum(variance, 0)),
            )
            put_feature(
                features,
                f"roll_{column}_nonzero_share_{window}d",
                real_mask,
                nonzero_sum / exposure,
            )

    return rolling_sums


def add_ewm_features(df, features):
    real_mask = df["day_index"].ge(0).to_numpy()
    real = df.loc[real_mask]

    for half_life in EWM_HALF_LIVES:
        alpha = 1 - 0.5 ** (1 / half_life)
        decay = 1 - alpha

        for column in MAIN_COLUMNS:
            scaled = (
                alpha
                * real[column].astype("float32")
                * decay ** (-real["day_index"])
            )
            cumulative = scaled.groupby(real["user_id"], sort=False).cumsum()
            ewm = decay ** real["day_index"] * cumulative

            put_feature(
                features,
                f"ewm_{column}_halflife_{half_life}d",
                real_mask,
                ewm,
            )


def add_trend_features(df, features, rolling_sums):
    real_mask = df["day_index"].ge(0).to_numpy()

    for column in MAIN_COLUMNS:
        recent_7 = rolling_sums[(column, 7)]
        previous_7 = rolling_sums[(column, 14)] - recent_7
        recent_30 = rolling_sums[(column, 30)]
        previous_30 = rolling_sums[(column, 60)] - recent_30

        put_feature(
            features,
            f"trend_{column}_7_vs_previous_7d",
            real_mask,
            bounded_change(recent_7, previous_7),
        )
        put_feature(
            features,
            f"trend_{column}_30_vs_previous_30d",
            real_mask,
            bounded_change(recent_30, previous_30),
        )


def add_recency_features(df, features):
    real_mask = df["day_index"].ge(0).to_numpy()
    real = df.loc[real_mask]

    for column in MAIN_COLUMNS:
        last_day = real["day_index"].where(real[column].gt(0))
        last_day = last_day.groupby(real["user_id"], sort=False).ffill()

        days_since = real["day_index"] - last_day
        never = last_day.isna().astype("float32")

        put_feature(
            features,
            f"days_since_{column}",
            real_mask,
            days_since.fillna(HISTORY_DAYS + 1),
        )
        put_feature(features, f"never_{column}", real_mask, never)


def add_ratio_features(df, features):
    real_mask = df["day_index"].ge(0).to_numpy()
    real = df.loc[real_mask].copy()
    rolled = rolling_table(real, RATIO_COLUMNS, 30)

    searches = features["roll_searches_sum_30d"][real_mask]
    carts = features["roll_to_cart_sum_30d"][real_mask]
    orders = features["roll_to_ord_sum_30d"][real_mask]
    gmv = features["roll_gmv_sum_30d"][real_mask]

    put_feature(
        features,
        "conv_search_to_cart_30d",
        real_mask,
        safe_divide(rolled["search_to_cart_sum"].fillna(0), searches),
    )
    put_feature(
        features,
        "conv_search_to_order_30d",
        real_mask,
        safe_divide(rolled["search_to_ord_sum"].fillna(0), searches),
    )
    put_feature(
        features,
        "conv_cart_to_order_30d",
        real_mask,
        safe_divide(orders, carts),
    )
    put_feature(
        features,
        "search_gmv_share_30d",
        real_mask,
        safe_divide(rolled["gmv_search_sum"].fillna(0), gmv),
    )
    put_feature(
        features,
        "search_order_share_30d",
        real_mask,
        safe_divide(rolled["search_to_ord_sum"].fillna(0), orders),
    )
    put_feature(
        features,
        "cat_order_share_30d",
        real_mask,
        safe_divide(rolled["cat_to_ord_sum"].fillna(0), orders),
    )
    put_feature(
        features,
        "gmv_per_order_30d",
        real_mask,
        safe_divide(gmv, orders),
    )


def build_features_for_batch(batch, cutoff):
    df = prepare_batch(batch, cutoff)
    features = {}

    features[CUTOFF_STATE_COLUMN] = df[CUTOFF_STATE_COLUMN].to_numpy()
    add_calendar_features(df, features)
    add_lag_features(df, features)
    rolling_sums = add_rolling_features(df, features)
    add_ewm_features(df, features)
    add_trend_features(df, features, rolling_sums)
    add_recency_features(df, features)
    add_ratio_features(df, features)

    feature_df = pd.DataFrame(features, index=df.index)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0).astype("float32")

    result = df.drop(columns=[CUTOFF_STATE_COLUMN])
    result = pd.concat([result, feature_df], axis=1)
    result = fix_dtypes(result)

    return result, len(feature_df.columns)


def fix_dtypes(df):
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["user_id"] = df["user_id"].astype("int64")
    df["day_index"] = df["day_index"].astype("int16")
    df["target"] = df["target"].astype("float32")

    feature_columns = [
        column
        for column in df.columns
        if column not in TECH_COLUMNS
    ]
    df[feature_columns] = df[feature_columns].astype("float32")

    return df


def read_batches(input_file):
    parquet_file = pq.ParquetFile(input_file)
    saved_tail = None

    for arrow_batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        batch = arrow_batch.to_pandas()

        if saved_tail is not None:
            batch = pd.concat([saved_tail, batch], ignore_index=True)

        last_user = batch["user_id"].iloc[-1]
        last_user_mask = batch["user_id"].eq(last_user)

        ready = batch.loc[~last_user_mask].copy()
        saved_tail = batch.loc[last_user_mask].copy()

        if len(ready) > 0:
            yield ready

    if saved_tail is not None:
        yield saved_tail


def create_feature_file(input_file, output_file, cutoff):
    writer = None
    rows_written = 0

    for batch_number, batch in enumerate(read_batches(input_file), start=1):
        batch_with_features, feature_count = build_features_for_batch(
            batch,
            cutoff,
        )
        table = pa.Table.from_pandas(batch_with_features, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                output_file,
                table.schema,
                compression="zstd",
            )
        else:
            table = table.cast(writer.schema)

        writer.write_table(table)
        rows_written += len(batch_with_features)

        print(
            f"batch={batch_number}, rows={rows_written:,}, "
            f"new_features={feature_count}"
        )

    if writer is not None:
        writer.close()

    print(f"Готово: {output_file}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for input_name, output_name, cutoff in FILES:
        create_feature_file(
            f"{INPUT_DIR}/{input_name}",
            f"{OUTPUT_DIR}/{output_name}",
            cutoff,
        )
