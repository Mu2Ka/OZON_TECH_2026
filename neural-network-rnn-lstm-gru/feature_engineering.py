

import os
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_DAYS = 180
LAG_COLUMNS = ["gmv", "searches", "to_cart", "to_ord"]
LAGS = [1, 7, 14, 30]

ROLLING_COLUMNS = ["gmv", "searches", "to_cart", "to_ord"]
ROLLING_WINDOWS = [7, 30, 90]

EWM_COLUMNS = ["gmv", "searches", "to_cart", "to_ord"]
EWM_HALF_LIVES = [7, 30]

RECENCY_COLUMNS = ["gmv", "searches", "to_cart", "to_ord"]

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "data_sequence"
OUTPUT_DIR = BASE_DIR / "data_sequence_features"
BATCH_SIZE = 250_000
OVERWRITE = False

REQUIRED_COLUMNS = [
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


def safe_divide(numerator, denominator):
    """Деление без inf и деления на ноль."""
    result = np.zeros(len(numerator), dtype="float64")
    np.divide(
        np.asarray(numerator, dtype="float64"),
        np.asarray(denominator, dtype="float64"),
        out=result,
        where=np.asarray(denominator) != 0,
    )
    return result


def bounded_change(recent, previous):
    """Изменение двух периодов примерно в диапазоне от -1 до 1."""
    return (
        (recent - previous)
        / (np.abs(recent) + np.abs(previous) + 1)
    )


def add_calendar_features(df):
    """Календарь и расстояние между записанными днями."""
    real_mask = df["day_index"] >= 0
    real = df.loc[real_mask]

    df["observed_day"] = real_mask.astype("float32")
    df["day_index_norm"] = 0.0
    df.loc[real_mask, "day_index_norm"] = (
        real["day_index"] / (HISTORY_DAYS - 1)
    )

    df["days_since_prev_observed"] = HISTORY_DAYS + 1
    gaps = real.groupby("user_id", sort=False)["day_index"].diff()
    first_rows = gaps.isna()
    gaps.loc[first_rows] = real.loc[first_rows, "day_index"] + 1
    df.loc[real_mask, "days_since_prev_observed"] = gaps

    day_of_week = real["event_date"].dt.dayofweek
    day_of_year = real["event_date"].dt.dayofyear

    df["calendar_is_weekend"] = 0.0
    df["calendar_day_of_week_sin"] = 0.0
    df["calendar_day_of_week_cos"] = 0.0
    df["calendar_day_of_year_sin"] = 0.0
    df["calendar_day_of_year_cos"] = 0.0

    df.loc[real_mask, "calendar_is_weekend"] = (day_of_week >= 5).astype(float)
    df.loc[real_mask, "calendar_day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    df.loc[real_mask, "calendar_day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )
    df.loc[real_mask, "calendar_day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    df.loc[real_mask, "calendar_day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )


def add_lag_features(df):
    """Лаги по календарным дням, а не по предыдущей строке."""
    real_mask = df["day_index"] >= 0
    real = df.loc[real_mask]

    key = pd.MultiIndex.from_frame(real[["user_id", "day_index"]])
    observed_lookup = pd.Series(1.0, index=key)

    for lag in LAGS:
        wanted_key = pd.MultiIndex.from_arrays(
            [real["user_id"], real["day_index"] - lag]
        )

        df[f"lag_observed_{lag}d"] = 0.0
        df.loc[real_mask, f"lag_observed_{lag}d"] = (
            observed_lookup.reindex(wanted_key).fillna(0).to_numpy()
        )

        for column in LAG_COLUMNS:
            value_lookup = pd.Series(real[column].to_numpy(), index=key)
            lag_values = value_lookup.reindex(wanted_key).fillna(0).to_numpy()
            df[f"lag_{column}_{lag}d"] = 0.0
            df.loc[real_mask, f"lag_{column}_{lag}d"] = lag_values


def calculate_rolling_table(real, window):
    """Суммы и квадраты внутри календарного окна."""
    temp = real[["user_id", "event_date", "day_index"]].copy()

    rolling_columns = []
    for column in ROLLING_COLUMNS:
        temp[f"{column}_sum"] = real[column].astype(float)
        temp[f"{column}_square"] = real[column].astype(float) ** 2
        temp[f"{column}_nonzero"] = (real[column] > 0).astype(float)
        rolling_columns.extend(
            [
                f"{column}_sum",
                f"{column}_square",
                f"{column}_nonzero",
            ]
        )

    rolling = (
        temp.set_index("event_date")
        .groupby("user_id", sort=False)[rolling_columns]
        .rolling(f"{window}D", closed="right")
        .sum()
        .reset_index()
    )

    rolling = real[["user_id", "event_date", "day_index"]].merge(
        rolling,
        on=["user_id", "event_date"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return rolling


def add_rolling_features(df):
    """Rolling суммы, стандартные отклонения и доли активных дней."""
    real_mask = df["day_index"] >= 0
    real = df.loc[real_mask].copy()

    # 14 и 60 нужны для сравнений с предыдущим периодом.
    all_windows = sorted(set(ROLLING_WINDOWS + [14, 60]))

    for window in all_windows:
        rolling = calculate_rolling_table(real, window)
        exposure = np.minimum(window, rolling["day_index"] + 1).clip(lower=1)

        for column in ROLLING_COLUMNS:
            rolling_sum = rolling[f"{column}_sum"].to_numpy()
            df[f"roll_{column}_sum_{window}d"] = 0.0
            df.loc[real_mask, f"roll_{column}_sum_{window}d"] = rolling_sum

            # Для 14 и 60 дней оставляем только сумму.
            if window not in ROLLING_WINDOWS:
                continue

            mean = rolling_sum / exposure
            variance = (
                rolling[f"{column}_square"].to_numpy() / exposure
                - mean ** 2
            )
            std = np.sqrt(np.maximum(variance, 0))
            nonzero_share = (
                rolling[f"{column}_nonzero"].to_numpy() / exposure
            )

            df[f"roll_{column}_std_{window}d"] = 0.0
            df[f"roll_{column}_nonzero_share_{window}d"] = 0.0
            df.loc[real_mask, f"roll_{column}_std_{window}d"] = std
            df.loc[
                real_mask,
                f"roll_{column}_nonzero_share_{window}d",
            ] = nonzero_share


def add_ewm_features(df):
    """Экспоненциальное среднее с затуханием в пропущенные дни."""
    real_mask = df["day_index"] >= 0
    real = df.loc[real_mask]

    for half_life in EWM_HALF_LIVES:
        alpha = 1 - 0.5 ** (1 / half_life)
        decay = 1 - alpha

        for column in EWM_COLUMNS:
            # Формула эквивалентна ежедневному EWM, где пропущенные дни равны 0.
            scaled = (
                alpha
                * real[column].astype(float)
                * decay ** (-real["day_index"])
            )
            cumulative = scaled.groupby(real["user_id"], sort=False).cumsum()
            values = decay ** real["day_index"] * cumulative

            name = f"ewm_{column}_halflife_{half_life}d"
            df[name] = 0.0
            df.loc[real_mask, name] = values.to_numpy()


def add_trend_features(df):
    """Последние 7/30 дней против предыдущих 7/30 дней."""
    for column in ROLLING_COLUMNS:
        recent_7 = df[f"roll_{column}_sum_7d"]
        previous_7 = df[f"roll_{column}_sum_14d"] - recent_7
        df[f"trend_{column}_7_vs_previous_7d"] = bounded_change(
            recent_7,
            previous_7,
        )

        recent_30 = df[f"roll_{column}_sum_30d"]
        previous_30 = df[f"roll_{column}_sum_60d"] - recent_30
        df[f"trend_{column}_30_vs_previous_30d"] = bounded_change(
            recent_30,
            previous_30,
        )


def add_recency_features(df):
    """Сколько дней прошло с последней активности каждого типа."""
    real_mask = df["day_index"] >= 0
    real = df.loc[real_mask]

    for column in RECENCY_COLUMNS:
        last_day = real["day_index"].where(real[column] > 0)
        last_day = last_day.groupby(real["user_id"], sort=False).ffill()

        days_since = real["day_index"] - last_day
        never = last_day.isna()

        df[f"days_since_{column}"] = HISTORY_DAYS + 1
        df[f"never_{column}"] = 1.0
        df.loc[real_mask, f"days_since_{column}"] = (
            days_since.fillna(HISTORY_DAYS + 1).to_numpy()
        )
        df.loc[real_mask, f"never_{column}"] = never.astype(float).to_numpy()


def add_ratio_features(df):
    """Конверсии и денежные отношения за последние 30 дней."""
    real_mask = df["day_index"] >= 0
    real = df.loc[real_mask].copy()

    ratio_columns = [
        "search_to_cart",
        "search_to_ord",
        "cat_to_ord",
        "gmv_search",
    ]

    temp = real[["user_id", "event_date"] + ratio_columns].copy()
    rolling = (
        temp.set_index("event_date")
        .groupby("user_id", sort=False)[ratio_columns]
        .rolling("30D", closed="right")
        .sum()
        .reset_index()
    )
    rolling = real[["user_id", "event_date"]].merge(
        rolling,
        on=["user_id", "event_date"],
        how="left",
        sort=False,
        validate="one_to_one",
    )

    searches = df.loc[real_mask, "roll_searches_sum_30d"].to_numpy()
    carts = df.loc[real_mask, "roll_to_cart_sum_30d"].to_numpy()
    orders = df.loc[real_mask, "roll_to_ord_sum_30d"].to_numpy()
    gmv = df.loc[real_mask, "roll_gmv_sum_30d"].to_numpy()

    ratio_features = {
        "conv_search_to_cart_30d": safe_divide(
            rolling["search_to_cart"], searches
        ),
        "conv_search_to_order_30d": safe_divide(
            rolling["search_to_ord"], searches
        ),
        "conv_cart_to_order_30d": safe_divide(orders, carts),
        "search_gmv_share_30d": safe_divide(rolling["gmv_search"], gmv),
        "search_order_share_30d": safe_divide(
            rolling["search_to_ord"], orders
        ),
        "cat_order_share_30d": safe_divide(rolling["cat_to_ord"], orders),
        "gmv_per_order_30d": safe_divide(gmv, orders),
    }

    for name, values in ratio_features.items():
        df[name] = 0.0
        df.loc[real_mask, name] = values


def build_sequence_features(sequence_df):
    """Построить все признаки только из текущего дня и прошлого."""
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in sequence_df.columns
    ]
    if missing:
        raise ValueError(f"Нет обязательных колонок: {missing}")

    df = sequence_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values(
        ["user_id", "day_index"],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_mask = (
        df.loc[df["day_index"] >= 0]
        .duplicated(["user_id", "day_index"])
    )
    if duplicate_mask.any():
        raise ValueError("Есть дубли user_id + day_index")

    old_columns = set(df.columns)

    add_calendar_features(df)
    add_lag_features(df)
    add_rolling_features(df)
    add_ewm_features(df)
    add_trend_features(df)
    add_recency_features(df)

    # После добавления большого числа колонок собираем DataFrame в один блок.
    df = df.copy()
    add_ratio_features(df)

    feature_columns = [
        column for column in df.columns
        if column not in old_columns
    ]
    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )

    return df, feature_columns


def read_user_batches(input_path, batch_size):
    """Читать parquet частями и не разрезать одного пользователя."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(input_path)
    last_user_rows = None

    for arrow_batch in parquet_file.iter_batches(batch_size=batch_size):
        batch = arrow_batch.to_pandas()

        if last_user_rows is not None:
            batch = pd.concat(
                [last_user_rows, batch],
                ignore_index=True,
                copy=False,
            )

        last_user = batch["user_id"].iloc[-1]
        last_user_mask = batch["user_id"] == last_user

        ready = batch.loc[~last_user_mask].copy()
        last_user_rows = batch.loc[last_user_mask].copy()

        if len(ready) > 0:
            yield ready

    if last_user_rows is not None:
        yield last_user_rows


def create_feature_file(
    input_file,
    output_file,
):
    """Создать parquet с исходными и новыми признаками."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_path = SOURCE_DIR / input_file
    output_path = OUTPUT_DIR / output_file

    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Файл уже существует: {output_path}. "
            "Поставьте OVERWRITE = True в начале файла для перезаписи."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.parquet")

    writer = None
    total_rows = 0

    try:
        for batch_number, batch in enumerate(
            read_user_batches(input_path, BATCH_SIZE),
            start=1,
        ):
            featured_batch, feature_columns = build_sequence_features(batch)
            table = pa.Table.from_pandas(featured_batch, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)
            total_rows += len(featured_batch)

            print(
                f"batch={batch_number}, rows={total_rows:,}, "
                f"new_features={len(feature_columns)}"
            )

        writer.close()
        writer = None
        os.replace(temp_path, output_path)

    finally:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()

    print(f"Готово: {output_path}")


if __name__ == "__main__":
    create_feature_file(
        "sequence_cutoff2026-02-13.parquet",
        "sequence_cutoff2026-02-13_features.parquet",
    )
