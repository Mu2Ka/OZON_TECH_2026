from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SOURCE = Path(r"C:\Users\myska\Downloads\train.parquet")
OUTPUT_DIR = Path("../data_classic")
BATCH_SIZE = 200_000

WINDOWS = (7, 14, 30, 60, 90, 180, 270)
RECENCY_WINDOW = 270
BLOCK_DAYS = 30
NUMBER_OF_BLOCKS = 9

SOURCE_COLUMNS = [
    "event_date",
    "user_id",
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

# Базовые агрегации внутри уже полностью собранной истории пользователя.
WINDOW_AGGREGATIONS = {
    "gmv": ("gmv", "sum"),
    "gmv_search": ("gmv_search", "sum"),
    "gmv_cat": ("gmv_cat", "sum"),
    "items_bought": ("to_ord", "sum"),
    "search_items_bought": ("search_to_ord", "sum"),
    "cat_items_bought": ("cat_to_ord", "sum"),
    "items_added_to_cart": ("to_cart", "sum"),
    "search_items_added_to_cart": ("search_to_cart", "sum"),
    "cat_items_added_to_cart": ("cat_to_cart", "sum"),
    "searches": ("searches", "sum"),
    "recorded_days": ("event_date", "size"),
    "active_days": ("active_day_flag", "sum"),
    "order_days": ("order_day_flag", "sum"),
    "cart_days": ("cart_day_flag", "sum"),
    "search_days": ("search_day_flag", "sum"),
    "cat_days": ("cat_day_flag", "sum"),
    "search_and_cat_days": ("search_and_cat_day_flag", "sum"),
    "search_to_cart_days": ("search_to_cart_day_flag", "sum"),
    "search_to_order_days": ("search_to_order_day_flag", "sum"),
    "cat_to_cart_days": ("cat_to_cart_day_flag", "sum"),
    "cat_to_order_days": ("cat_to_order_day_flag", "sum"),
    "cart_without_order_days": ("cart_without_order_day_flag", "sum"),
    "active_without_order_days": ("active_without_order_day_flag", "sum"),
    # Суммы квадратов нужны для стандартных отклонений по календарным дням.
    "gmv_square_sum": ("gmv_square", "sum"),
    "items_bought_square_sum": ("items_bought_square", "sum"),
    "searches_square_sum": ("searches_square", "sum"),
    "max_daily_gmv": ("gmv", "max"),
    "max_daily_items_bought": ("to_ord", "max"),
    "max_daily_items_added_to_cart": ("to_cart", "max"),
    "max_daily_searches": ("searches", "max"),
}

BLOCK_AGGREGATIONS = {
    "gmv": ("gmv", "sum"),
    "items_bought": ("to_ord", "sum"),
    "order_days": ("order_day_flag", "sum"),
    "items_added_to_cart": ("to_cart", "sum"),
    "searches": ("searches", "sum"),
    "active_days": ("active_day_flag", "sum"),
}

EVENT_FLAGS = {
    "activity": "active_day_flag",
    "order": "order_day_flag",
    "cart": "cart_day_flag",
    "search": "search_day_flag",
    "cat": "cat_day_flag",
    "search_cart": "search_to_cart_day_flag",
    "search_order": "search_to_order_day_flag",
    "cat_cart": "cat_to_cart_day_flag",
    "cat_order": "cat_to_order_day_flag",
}


def safe_ratio(numerator, denominator):
    """Делит Series на Series и возвращает 0 при нулевом знаменателе."""

    valid_denominator = denominator.where(denominator.ne(0))
    return numerator.div(valid_denominator).fillna(0)


def bounded_change(recent, previous):
    """Симметричное изменение в диапазоне примерно от -1 до 1."""

    return (recent - previous).div(
        recent.abs() + previous.abs() + 1
    )


def add_daily_flags(data):
    """Добавляет служебные дневные флаги и моменты для агрегации."""

    data = data.copy()

    data["search_day_flag"] = data["search"].gt(0).astype("int8")
    data["cat_day_flag"] = data["cat"].gt(0).astype("int8")
    data["order_day_flag"] = data["to_ord"].gt(0).astype("int8")
    data["cart_day_flag"] = data["to_cart"].gt(0).astype("int8")

    data["search_to_cart_day_flag"] = (
        data["has_search_to_cart"].gt(0).astype("int8")
    )
    data["search_to_order_day_flag"] = (
        data["has_search_to_ord"].gt(0).astype("int8")
    )
    data["cat_to_cart_day_flag"] = (
        data["has_cat_to_cart"].gt(0).astype("int8")
    )
    data["cat_to_order_day_flag"] = (
        data["has_cat_to_ord"].gt(0).astype("int8")
    )

    data["active_day_flag"] = (
            data["search_day_flag"].gt(0)
            | data["cat_day_flag"].gt(0)
            | data["cart_day_flag"].gt(0)
            | data["order_day_flag"].gt(0)
            | data["searches"].gt(0)
    ).astype("int8")

    data["search_and_cat_day_flag"] = (
            data["search_day_flag"].gt(0)
            & data["cat_day_flag"].gt(0)
    ).astype("int8")
    data["cart_without_order_day_flag"] = (
            data["cart_day_flag"].gt(0)
            & data["order_day_flag"].eq(0)
    ).astype("int8")
    data["active_without_order_day_flag"] = (
            data["active_day_flag"].gt(0)
            & data["order_day_flag"].eq(0)
    ).astype("int8")

    data["gmv_square"] = data["gmv"].pow(2)
    data["items_bought_square"] = data["to_ord"].pow(2)
    data["searches_square"] = data["searches"].pow(2)

    day_number = (
            data["event_date"] - pd.Timestamp("2025-01-01")
    ).dt.days
    data["calendar_week"] = (day_number // 7).astype("int16")
    data["calendar_month"] = (
            data["event_date"].dt.year * 12
            + data["event_date"].dt.month
    ).astype("int16")
    data["weekend_flag"] = (
            data["event_date"].dt.dayofweek >= 5
    ).astype("int8")

    return data


def iter_complete_user_chunks(parquet):
    """Не разрезает историю пользователя между двумя обрабатываемыми батчами."""

    carry = None

    for batch_number, batch in enumerate(
            parquet.iter_batches(
                batch_size=BATCH_SIZE,
                columns=SOURCE_COLUMNS,
            )
    ):
        chunk = batch.to_pandas()
        chunk["event_date"] = pd.to_datetime(chunk["event_date"])

        if carry is not None:
            chunk = pd.concat(
                [carry, chunk],
                ignore_index=True,
                copy=False,
            )

        if not chunk["user_id"].is_monotonic_increasing:
            raise ValueError(
                "train.parquet должен быть отсортирован по user_id"
            )

        last_user = chunk["user_id"].iloc[-1]
        last_user_mask = chunk["user_id"].eq(last_user)

        carry = chunk.loc[last_user_mask].copy()
        complete_chunk = chunk.loc[~last_user_mask].copy()

        if not complete_chunk.empty:
            yield batch_number, complete_chunk

    if carry is not None and not carry.empty:
        yield batch_number + 1, carry


def aggregate_window(data, cutoff, days):
    start = cutoff - pd.Timedelta(days=days - 1)
    window = data.loc[
        data["event_date"].between(start, cutoff)
    ]

    features = (
        window.groupby("user_id", sort=False)
        .agg(**WINDOW_AGGREGATIONS)
        .add_suffix(f"_{days}d")
    )
    return features


def aggregate_30d_blocks(data, cutoff):
    """Девять непересекающихся блоков, совпадающих с горизонтом target."""

    result = None

    for block_number in range(NUMBER_OF_BLOCKS):
        block_end = cutoff - pd.Timedelta(
            days=block_number * BLOCK_DAYS
        )
        block_start = block_end - pd.Timedelta(
            days=BLOCK_DAYS - 1
        )

        block = data.loc[
            data["event_date"].between(block_start, block_end)
        ]
        block_features = block.groupby(
            "user_id",
            sort=False,
        ).agg(**BLOCK_AGGREGATIONS)
        block_features = block_features.rename(
            columns={
                column: f"{column}_block_{block_number}_30d"
                for column in block_features.columns
            }
        )

        if result is None:
            result = block_features
        else:
            result = result.join(block_features, how="outer")

    return result


def add_timing_and_regularity_features(result, data, cutoff):
    """RFM-recency, возраст истории, интервалы и регулярность."""

    history_start = cutoff - pd.Timedelta(
        days=RECENCY_WINDOW - 1
    )
    history = data.loc[
        data["event_date"].between(history_start, cutoff)
    ].copy()
    history["days_ago"] = (
            cutoff - history["event_date"]
    ).dt.days
    feature_parts = []

    for event_name, flag_column in EVENT_FLAGS.items():
        events = history.loc[
            history[flag_column].gt(0),
            ["user_id", "days_ago"],
        ]
        timing = events.groupby("user_id", sort=False)[
            "days_ago"
        ].agg(["min", "max"])

        last_column = f"days_since_last_{event_name}_{RECENCY_WINDOW}d"
        first_column = f"days_since_first_{event_name}_{RECENCY_WINDOW}d"
        span_column = f"{event_name}_span_{RECENCY_WINDOW}d"
        never_column = f"never_{event_name}_{RECENCY_WINDOW}d"

        timing = timing.rename(
            columns={"min": last_column, "max": first_column}
        )
        timing[never_column] = timing[last_column].isna().astype("int8")
        timing[last_column] = timing[last_column].fillna(
            RECENCY_WINDOW + 1
        )
        timing[first_column] = timing[first_column].fillna(0)
        timing[span_column] = (
                timing[first_column] - timing[last_column]
        ).clip(lower=0)
        feature_parts.append(timing)

    for event_name in ("activity", "order", "cart"):
        flag_column = EVENT_FLAGS[event_name]
        events = history.loc[
            history[flag_column].gt(0),
            ["user_id", "event_date"],
        ].copy()
        events["gap"] = events.groupby(
            "user_id",
            sort=False,
        )["event_date"].diff().dt.days

        gap_features = events.groupby(
            "user_id",
            sort=False,
        )["gap"].agg(["mean", "median", "std", "max", "last"])
        gap_features = gap_features.rename(
            columns={
                "mean": f"{event_name}_gap_mean_{RECENCY_WINDOW}d",
                "median": f"{event_name}_gap_median_{RECENCY_WINDOW}d",
                "std": f"{event_name}_gap_std_{RECENCY_WINDOW}d",
                "max": f"{event_name}_gap_max_{RECENCY_WINDOW}d",
                "last": f"{event_name}_gap_last_{RECENCY_WINDOW}d",
            }
        )
        feature_parts.append(gap_features)

        if event_name == "order":
            gap_quantiles = events.groupby(
                "user_id",
                sort=False,
            )["gap"].quantile([0.25, 0.75, 0.9]).unstack()
            gap_quantiles = gap_quantiles.rename(
                columns={
                    0.25: f"order_gap_q25_{RECENCY_WINDOW}d",
                    0.75: f"order_gap_q75_{RECENCY_WINDOW}d",
                    0.9: f"order_gap_q90_{RECENCY_WINDOW}d",
                }
            )
            feature_parts.append(gap_quantiles)

        events["new_streak"] = events.groupby(
            "user_id",
            sort=False,
        )["event_date"].diff().dt.days.ne(1)
        events["streak_id"] = events.groupby(
            "user_id",
            sort=False,
        )["new_streak"].cumsum()
        longest_streak = (
            events.groupby(
                ["user_id", "streak_id"],
                sort=False,
            )
            .size()
            .groupby(level=0)
            .max()
            .rename(f"{event_name}_longest_streak_{RECENCY_WINDOW}d")
        )
        feature_parts.append(longest_streak.to_frame())

    for event_name in ("activity", "order", "cart", "search"):
        flag_column = EVENT_FLAGS[event_name]
        events = history.loc[history[flag_column].gt(0)]

        weeks = events.groupby("user_id", sort=False)[
            "calendar_week"
        ].nunique().rename(
            f"{event_name}_weeks_{RECENCY_WINDOW}d"
        )
        feature_parts.append(weeks.to_frame())

    for event_name in ("activity", "order"):
        flag_column = EVENT_FLAGS[event_name]
        events = history.loc[history[flag_column].gt(0)]

        months = events.groupby("user_id", sort=False)[
            "calendar_month"
        ].nunique().rename(
            f"{event_name}_months_{RECENCY_WINDOW}d"
        )
        feature_parts.append(months.to_frame())

    for days in (90, 180, 270):
        start = cutoff - pd.Timedelta(days=days - 1)
        window = history.loc[
            history["event_date"].between(start, cutoff)
        ].copy()
        window["weekend_gmv"] = window["gmv"] * window["weekend_flag"]
        window["weekend_items_bought"] = (
                window["to_ord"] * window["weekend_flag"]
        )
        window["weekend_active_day"] = (
                window["active_day_flag"] * window["weekend_flag"]
        )

        weekend = window.groupby("user_id", sort=False).agg(
            **{
                f"weekend_gmv_{days}d": ("weekend_gmv", "sum"),
                f"weekend_items_bought_{days}d": (
                    "weekend_items_bought",
                    "sum",
                ),
                f"weekend_active_days_{days}d": (
                    "weekend_active_day",
                    "sum",
                ),
            }
        )
        feature_parts.append(weekend)

    order_rows = history.loc[
        history["order_day_flag"].gt(0),
        [
            "user_id",
            "event_date",
            "gmv",
            "gmv_search",
            "gmv_cat",
            "to_ord",
        ],
    ]
    last_order = (
        order_rows.groupby("user_id", sort=False)
        .tail(1)
        .set_index("user_id")
        .rename(
            columns={
                "gmv": f"last_order_day_gmv_{RECENCY_WINDOW}d",
                "gmv_search": (
                    f"last_order_day_gmv_search_{RECENCY_WINDOW}d"
                ),
                "gmv_cat": f"last_order_day_gmv_cat_{RECENCY_WINDOW}d",
                "to_ord": (
                    f"last_order_day_items_bought_{RECENCY_WINDOW}d"
                ),
            }
        )
        .drop(columns="event_date")
    )
    feature_parts.append(last_order)

    order_day_distribution = order_rows.groupby(
        "user_id",
        sort=False,
    )["gmv"].agg(["median", "mean", "std"])
    order_day_distribution = order_day_distribution.rename(
        columns={
            "median": f"order_day_gmv_median_{RECENCY_WINDOW}d",
            "mean": f"order_day_gmv_mean_{RECENCY_WINDOW}d",
            "std": f"order_day_gmv_std_{RECENCY_WINDOW}d",
        }
    )
    feature_parts.append(order_day_distribution)

    recent_orders = order_rows.groupby(
        "user_id",
        sort=False,
    ).tail(3).copy()
    recent_orders["recent_order_rank"] = (
        recent_orders.groupby("user_id", sort=False)
        .cumcount(ascending=False)
        .add(1)
    )

    recent_order_gmv = recent_orders.pivot(
        index="user_id",
        columns="recent_order_rank",
        values="gmv",
    ).reindex(columns=[1, 2, 3])
    recent_order_gmv.columns = [
        f"last_{rank}_order_day_gmv_{RECENCY_WINDOW}d"
        for rank in recent_order_gmv.columns
    ]
    feature_parts.append(recent_order_gmv)

    recent_order_items = recent_orders.pivot(
        index="user_id",
        columns="recent_order_rank",
        values="to_ord",
    ).reindex(columns=[1, 2, 3])
    recent_order_items.columns = [
        f"last_{rank}_order_day_items_bought_{RECENCY_WINDOW}d"
        for rank in recent_order_items.columns
    ]
    feature_parts.append(recent_order_items)

    order_rows_with_phase = order_rows.copy()
    month_phase_labels = ["1_7", "8_15", "16_23", "24_end"]
    day_of_month = order_rows_with_phase["event_date"].dt.day
    order_rows_with_phase["month_phase"] = pd.cut(
        day_of_month,
        bins=[0, 7, 15, 23, 31],
        labels=month_phase_labels,
        include_lowest=True,
    )
    phase_counts = pd.crosstab(
        order_rows_with_phase["user_id"],
        order_rows_with_phase["month_phase"],
    ).reindex(columns=month_phase_labels, fill_value=0)
    phase_counts.columns = [
        f"order_days_month_phase_{phase}_{RECENCY_WINDOW}d"
        for phase in phase_counts.columns
    ]
    feature_parts.append(phase_counts)

    phase_gmv = order_rows_with_phase.pivot_table(
        index="user_id",
        columns="month_phase",
        values="gmv",
        aggfunc="sum",
        observed = False,
    ).reindex(columns=month_phase_labels)
    phase_gmv.columns = [
        f"gmv_month_phase_{phase}_{RECENCY_WINDOW}d"
        for phase in phase_gmv.columns
    ]
    feature_parts.append(phase_gmv)

    last_order_date = order_rows.groupby(
        "user_id",
        sort=False,
    )["event_date"].max()
    after_last_order = history.loc[
        history["event_date"].gt(
            history["user_id"].map(last_order_date)
        )
    ].copy()
    after_last_order["cart_value"] = after_last_order["to_cart"]
    after_last_order["active_value"] = after_last_order["active_day_flag"]
    after_last_order_features = after_last_order.groupby(
        "user_id",
        sort=False,
    ).agg(
        **{
            f"searches_after_last_order_{RECENCY_WINDOW}d": (
                "searches",
                "sum",
            ),
            f"items_added_to_cart_after_last_order_{RECENCY_WINDOW}d": (
                "cart_value",
                "sum",
            ),
            f"search_days_after_last_order_{RECENCY_WINDOW}d": (
                "search_day_flag",
                "sum",
            ),
            f"cart_days_after_last_order_{RECENCY_WINDOW}d": (
                "cart_day_flag",
                "sum",
            ),
            f"active_days_after_last_order_{RECENCY_WINDOW}d": (
                "active_value",
                "sum",
            ),
        }
    )
    feature_parts.append(after_last_order_features)

    decay_source_columns = {
        "gmv": "gmv",
        "items_bought": "to_ord",
        "items_added_to_cart": "to_cart",
        "searches": "searches",
        "activity": "active_day_flag",
    }
    for half_life in (7, 14, 30, 60, 90):
        weights = np.power(
            0.5,
            history["days_ago"] / half_life,
        )
        decay_data = pd.DataFrame(
            {"user_id": history["user_id"]}
        )
        aggregations = {}
        for feature_name, source_column in decay_source_columns.items():
            weighted_column = f"weighted_{feature_name}"
            decay_data[weighted_column] = (
                    history[source_column] * weights
            )
            aggregations[
                f"decay_{feature_name}_halflife_{half_life}d"
            ] = (weighted_column, "sum")

        decay_features = decay_data.groupby(
            "user_id",
            sort=False,
        ).agg(**aggregations)
        feature_parts.append(decay_features)

    timing_features = pd.concat(feature_parts, axis=1, copy=False)
    result = pd.concat(
        [result, timing_features],
        axis=1,
        copy=False,
    )

    # Эти строки отсутствуют в timing_features у пользователей без события.
    for event_name in EVENT_FLAGS:
        last_column = f"days_since_last_{event_name}_{RECENCY_WINDOW}d"
        first_column = f"days_since_first_{event_name}_{RECENCY_WINDOW}d"
        span_column = f"{event_name}_span_{RECENCY_WINDOW}d"
        never_column = f"never_{event_name}_{RECENCY_WINDOW}d"

        result[never_column] = result[never_column].fillna(1)
        result[last_column] = result[last_column].fillna(
            RECENCY_WINDOW + 1
        )
        result[first_column] = result[first_column].fillna(0)
        result[span_column] = result[span_column].fillna(0)

    return result


def build_chunk_features(data, cutoff, make_target):
    user_index = pd.Index(
        data["user_id"].drop_duplicates(),
        name="user_id",
    )
    relevant_start = cutoff - pd.Timedelta(days=RECENCY_WINDOW - 1)
    relevant_end = (
        cutoff + pd.Timedelta(days=30)
        if make_target
        else cutoff
    )
    data = data.loc[
        data["event_date"].between(relevant_start, relevant_end)
    ]
    data = add_daily_flags(data)

    feature_parts = []
    for days in WINDOWS:
        feature_parts.append(aggregate_window(data, cutoff, days))
    feature_parts.append(aggregate_30d_blocks(data, cutoff))

    result = pd.concat(
        feature_parts,
        axis=1,
        copy=False,
    )
    result = result.reindex(user_index)
    result = add_timing_and_regularity_features(
        result,
        data,
        cutoff,
    )

    if make_target:
        target_start = cutoff + pd.Timedelta(days=1)
        target_end = cutoff + pd.Timedelta(days=30)
        target = (
            data.loc[
                data["event_date"].between(target_start, target_end)
            ]
            .groupby("user_id", sort=False)["gmv"]
            .sum()
            .rename("target")
        )
        result = result.join(target, how="left")

    # Небольшая копия части устраняет фрагментацию после множества join/concat.
    result = result.copy()
    return result.fillna(0).reset_index()


def add_block_statistics(total, derived):
    """Форма, стабильность и zero-inflation прошлых 30-дневных GMV."""

    recent_weights = np.power(
        0.75,
        np.arange(NUMBER_OF_BLOCKS, dtype="float32"),
    )
    recent_weights /= recent_weights.sum()
    slope_weights = np.arange(
        NUMBER_OF_BLOCKS // 2,
        -(NUMBER_OF_BLOCKS // 2) - 1,
        -1,
        dtype="float32",
    )
    slope_denominator = np.square(slope_weights).sum()

    for metric in BLOCK_AGGREGATIONS:
        columns = [
            f"{metric}_block_{block}_30d"
            for block in range(NUMBER_OF_BLOCKS)
        ]
        values = total[columns].to_numpy(dtype="float32", copy=False)
        nonzero = values > 0
        nonzero_count = nonzero.sum(axis=1)

        derived[f"{metric}_block_mean"] = values.mean(axis=1)
        derived[f"{metric}_block_std"] = values.std(axis=1)
        derived[f"{metric}_block_min"] = values.min(axis=1)
        derived[f"{metric}_block_max"] = values.max(axis=1)
        derived[f"{metric}_nonzero_blocks"] = nonzero_count
        derived[f"{metric}_zero_block_share"] = 1 - (
                nonzero_count / NUMBER_OF_BLOCKS
        )
        derived[f"{metric}_positive_block_mean"] = (
                values.sum(axis=1)
                / np.maximum(nonzero_count, 1)
        )
        derived[f"{metric}_block_slope"] = (
                values @ slope_weights / slope_denominator
        )
        derived[f"{metric}_block_ewm"] = values @ recent_weights
        derived[f"{metric}_recent_to_block_mean"] = (
                (values[:, 0] + 1)
                / (values.mean(axis=1) + 1)
        )

        zero = ~nonzero
        current_zero_run = np.ones(len(total), dtype="int8")
        current_zero_run[~zero[:, 0]] = 0
        for block in range(1, NUMBER_OF_BLOCKS):
            current_zero_run += (
                zero[:, : block + 1].all(axis=1)
            ).astype("int8")
        derived[f"{metric}_recent_zero_block_run"] = current_zero_run


def add_derived_features(total):
    """Считает отношения только после получения полной истории пользователя."""

    derived = {}
    internal_columns = []

    for days in WINDOWS:
        gmv = total[f"gmv_{days}d"]
        items = total[f"items_bought_{days}d"]
        carts = total[f"items_added_to_cart_{days}d"]
        searches = total[f"searches_{days}d"]
        active_days = total[f"active_days_{days}d"]
        order_days = total[f"order_days_{days}d"]
        cart_days = total[f"cart_days_{days}d"]

        derived[f"gmv_per_item_{days}d"] = safe_ratio(gmv, items)
        derived[f"gmv_per_order_day_{days}d"] = safe_ratio(
            gmv,
            order_days,
        )
        derived[f"items_per_order_day_{days}d"] = safe_ratio(
            items,
            order_days,
        )
        derived[f"items_per_active_day_{days}d"] = safe_ratio(
            items,
            active_days,
        )
        derived[f"cart_to_order_conversion_{days}d"] = safe_ratio(
            items,
            carts,
        )
        derived[f"search_cart_to_order_conversion_{days}d"] = safe_ratio(
            total[f"search_items_bought_{days}d"],
            total[f"search_items_added_to_cart_{days}d"],
        )
        derived[f"cat_cart_to_order_conversion_{days}d"] = safe_ratio(
            total[f"cat_items_bought_{days}d"],
            total[f"cat_items_added_to_cart_{days}d"],
        )
        derived[f"search_query_to_cart_conversion_{days}d"] = safe_ratio(
            total[f"search_items_added_to_cart_{days}d"],
            searches,
        )
        derived[f"search_query_to_order_conversion_{days}d"] = safe_ratio(
            total[f"search_items_bought_{days}d"],
            searches,
        )
        derived[f"searches_per_search_day_{days}d"] = safe_ratio(
            searches,
            total[f"search_days_{days}d"],
        )

        derived[f"recorded_zero_days_{days}d"] = (
                total[f"recorded_days_{days}d"] - active_days
        ).clip(lower=0)
        derived[f"order_days_per_active_day_{days}d"] = safe_ratio(
            order_days,
            active_days,
        )
        derived[f"active_without_order_share_{days}d"] = safe_ratio(
            total[f"active_without_order_days_{days}d"],
            active_days,
        )
        derived[f"cart_without_order_share_{days}d"] = safe_ratio(
            total[f"cart_without_order_days_{days}d"],
            cart_days,
        )
        derived[f"search_gmv_share_{days}d"] = safe_ratio(
            total[f"gmv_search_{days}d"],
            gmv,
        )
        derived[f"search_items_share_{days}d"] = safe_ratio(
            total[f"search_items_bought_{days}d"],
            items,
        )
        derived[f"search_cart_share_{days}d"] = safe_ratio(
            total[f"search_items_added_to_cart_{days}d"],
            carts,
        )

        calendar_gmv_mean = gmv / days
        calendar_gmv_variance = (
                total[f"gmv_square_sum_{days}d"] / days
                - calendar_gmv_mean.pow(2)
        ).clip(lower=0)
        calendar_items_mean = items / days
        calendar_items_variance = (
                total[f"items_bought_square_sum_{days}d"] / days
                - calendar_items_mean.pow(2)
        ).clip(lower=0)
        calendar_searches_mean = searches / days
        calendar_searches_variance = (
                total[f"searches_square_sum_{days}d"] / days
                - calendar_searches_mean.pow(2)
        ).clip(lower=0)

        gmv_std = np.sqrt(calendar_gmv_variance)
        derived[f"daily_gmv_std_{days}d"] = gmv_std
        derived[f"daily_gmv_cv_{days}d"] = safe_ratio(
            gmv_std,
            calendar_gmv_mean + 1,
        )
        derived[f"daily_items_bought_std_{days}d"] = np.sqrt(
            calendar_items_variance
        )
        derived[f"daily_searches_std_{days}d"] = np.sqrt(
            calendar_searches_variance
        )
        derived[f"max_daily_gmv_share_{days}d"] = safe_ratio(
            total[f"max_daily_gmv_{days}d"],
            gmv,
        )

        internal_columns.extend(
            [
                f"gmv_square_sum_{days}d",
                f"items_bought_square_sum_{days}d",
                f"searches_square_sum_{days}d",
            ]
        )

    for days in (90, 180, 270):
        derived[f"weekend_gmv_share_{days}d"] = safe_ratio(
            total[f"weekend_gmv_{days}d"],
            total[f"gmv_{days}d"],
        )
        derived[f"weekend_items_share_{days}d"] = safe_ratio(
            total[f"weekend_items_bought_{days}d"],
            total[f"items_bought_{days}d"],
        )
        derived[f"weekend_activity_share_{days}d"] = safe_ratio(
            total[f"weekend_active_days_{days}d"],
            total[f"active_days_{days}d"],
        )

    trend_metrics = (
        "gmv",
        "items_bought",
        "items_added_to_cart",
        "searches",
        "active_days",
        "order_days",
    )
    for metric in trend_metrics:
        recent_7 = total[f"{metric}_7d"]
        previous_7 = (
                total[f"{metric}_14d"] - recent_7
        ).clip(lower=0)
        recent_30 = total[f"{metric}_30d"]
        previous_30 = (
                total[f"{metric}_60d"] - recent_30
        ).clip(lower=0)
        recent_90 = total[f"{metric}_90d"]
        previous_90 = (
                total[f"{metric}_180d"] - recent_90
        ).clip(lower=0)
        older_150 = (
                total[f"{metric}_180d"] - recent_30
        ).clip(lower=0)

        derived[f"{metric}_trend_7_vs_previous_7"] = bounded_change(
            recent_7,
            previous_7,
        )
        derived[f"{metric}_trend_30_vs_previous_30"] = bounded_change(
            recent_30,
            previous_30,
        )
        derived[f"{metric}_trend_90_vs_previous_90"] = bounded_change(
            recent_90,
            previous_90,
        )
        derived[f"{metric}_recent_30_share_180"] = safe_ratio(
            recent_30,
            total[f"{metric}_180d"],
        )
        derived[f"{metric}_rate_trend_30_vs_older_150"] = bounded_change(
            recent_30 / 30,
            older_150 / 150,
        )

    # Сохраняем знакомые имена текущих trend-признаков.
    derived["items_trend_30_180"] = safe_ratio(
        total["items_bought_30d"] + 1,
        total["items_bought_180d"] / 6 + 1,
    )
    derived["gmv_trend_30_180"] = safe_ratio(
        total["gmv_30d"] + 1,
        total["gmv_180d"] / 6 + 1,
    )
    derived["items_trend_90_180"] = safe_ratio(
        total["items_bought_90d"] + 1,
        total["items_bought_180d"] / 2 + 1,
    )
    derived["gmv_trend_90_180"] = safe_ratio(
        total["gmv_90d"] + 1,
        total["gmv_180d"] / 2 + 1,
    )

    search_gmv_share_30 = safe_ratio(
        total["gmv_search_30d"],
        total["gmv_30d"],
    )
    search_gmv_share_180 = safe_ratio(
        total["gmv_search_180d"],
        total["gmv_180d"],
    )
    total_conversion_30 = safe_ratio(
        total["items_bought_30d"],
        total["items_added_to_cart_30d"],
    )
    total_conversion_180 = safe_ratio(
        total["items_bought_180d"],
        total["items_added_to_cart_180d"],
    )
    search_conversion_30 = safe_ratio(
        total["search_items_bought_30d"],
        total["search_items_added_to_cart_30d"],
    )
    search_conversion_180 = safe_ratio(
        total["search_items_bought_180d"],
        total["search_items_added_to_cart_180d"],
    )
    cat_conversion_30 = safe_ratio(
        total["cat_items_bought_30d"],
        total["cat_items_added_to_cart_30d"],
    )
    cat_conversion_180 = safe_ratio(
        total["cat_items_bought_180d"],
        total["cat_items_added_to_cart_180d"],
    )
    derived["search_gmv_share_change_30_vs_180"] = (
            search_gmv_share_30 - search_gmv_share_180
    )
    derived["cart_to_order_conversion_change_30_vs_180"] = (
            total_conversion_30 - total_conversion_180
    )
    derived["search_conversion_change_30_vs_180"] = (
            search_conversion_30 - search_conversion_180
    )
    derived["cat_conversion_change_30_vs_180"] = (
            cat_conversion_30 - cat_conversion_180
    )
    derived["channel_conversion_gap_30d"] = (
            search_conversion_30 - cat_conversion_30
    )

    for event_name in ("activity", "order", "cart"):
        mean_gap = total[f"{event_name}_gap_mean_{RECENCY_WINDOW}d"]
        std_gap = total[f"{event_name}_gap_std_{RECENCY_WINDOW}d"]
        derived[f"{event_name}_gap_cv_{RECENCY_WINDOW}d"] = safe_ratio(
            std_gap,
            mean_gap + 1,
        )
        derived[f"{event_name}_burstiness_{RECENCY_WINDOW}d"] = (
                (std_gap - mean_gap)
                / (std_gap + mean_gap + 1)
        )

    order_age = total[f"days_since_first_order_{RECENCY_WINDOW}d"]
    order_recency = total[f"days_since_last_order_{RECENCY_WINDOW}d"]
    order_frequency = total[f"order_days_{RECENCY_WINDOW}d"]
    order_mean_gap = total[f"order_gap_mean_{RECENCY_WINDOW}d"]
    order_median_gap = total[f"order_gap_median_{RECENCY_WINDOW}d"]
    order_last_gap = total[f"order_gap_last_{RECENCY_WINDOW}d"]
    order_gap_q25 = total[f"order_gap_q25_{RECENCY_WINDOW}d"]
    order_gap_q75 = total[f"order_gap_q75_{RECENCY_WINDOW}d"]
    order_gap_q90 = total[f"order_gap_q90_{RECENCY_WINDOW}d"]
    has_order_history = (
            1 - total[f"never_order_{RECENCY_WINDOW}d"]
    )

    derived[f"order_overdue_ratio_{RECENCY_WINDOW}d"] = (
            safe_ratio(order_recency, order_median_gap + 1)
            * has_order_history
    )
    derived[f"order_days_to_expected_{RECENCY_WINDOW}d"] = (
            (order_median_gap - order_recency)
            * has_order_history
    )
    derived[f"order_overdue_days_{RECENCY_WINDOW}d"] = (
            (order_recency - order_median_gap).clip(lower=0)
            * has_order_history
    )
    derived[f"order_expected_within_30d_{RECENCY_WINDOW}d"] = (
            (order_recency + 30 >= order_median_gap)
            & has_order_history.astype(bool)
    ).astype("int8")
    derived[f"order_cycle_phase_{RECENCY_WINDOW}d"] = (
            safe_ratio(order_recency, order_median_gap + 1)
            .clip(upper=5)
            * has_order_history
    )
    derived[f"order_last_gap_to_median_{RECENCY_WINDOW}d"] = (
            safe_ratio(order_last_gap, order_median_gap + 1)
            * has_order_history
    )
    derived[f"order_last_gap_to_mean_{RECENCY_WINDOW}d"] = (
            safe_ratio(order_last_gap, order_mean_gap + 1)
            * has_order_history
    )
    derived[f"order_gap_iqr_{RECENCY_WINDOW}d"] = (
            (order_gap_q75 - order_gap_q25).clip(lower=0)
            * has_order_history
    )
    derived[f"order_gap_q90_to_median_{RECENCY_WINDOW}d"] = (
            safe_ratio(order_gap_q90, order_median_gap + 1)
            * has_order_history
    )
    derived[f"expected_orders_next_30d_{RECENCY_WINDOW}d"] = (
            safe_ratio(
                pd.Series(30, index=total.index),
                order_median_gap + 1,
            )
            * has_order_history
    )
    derived[f"expected_orders_next_30d_with_overdue_{RECENCY_WINDOW}d"] = (
            safe_ratio(
                30 + derived[f"order_overdue_days_{RECENCY_WINDOW}d"],
                order_median_gap + 1,
            )
            * has_order_history
    )

    last_1_gmv = total[f"last_1_order_day_gmv_{RECENCY_WINDOW}d"]
    last_2_gmv = total[f"last_2_order_day_gmv_{RECENCY_WINDOW}d"]
    last_3_gmv = total[f"last_3_order_day_gmv_{RECENCY_WINDOW}d"]
    previous_two_gmv_mean = (last_2_gmv + last_3_gmv) / 2
    derived[f"last_order_gmv_to_previous_two_mean_{RECENCY_WINDOW}d"] = (
            safe_ratio(last_1_gmv, previous_two_gmv_mean + 1)
            * has_order_history
    )
    derived[f"last_order_gmv_trend_vs_previous_two_{RECENCY_WINDOW}d"] = (
            bounded_change(last_1_gmv, previous_two_gmv_mean)
            * has_order_history
    )

    derived[f"searches_after_last_order_per_day_{RECENCY_WINDOW}d"] = (
            safe_ratio(
                total[f"searches_after_last_order_{RECENCY_WINDOW}d"],
                order_recency + 1,
            )
            * has_order_history
    )
    derived[f"cart_items_after_last_order_per_day_{RECENCY_WINDOW}d"] = (
            safe_ratio(
                total[
                    f"items_added_to_cart_after_last_order_{RECENCY_WINDOW}d"
                ],
                order_recency + 1,
            )
            * has_order_history
    )
    derived[f"active_share_after_last_order_{RECENCY_WINDOW}d"] = (
            safe_ratio(
                total[f"active_days_after_last_order_{RECENCY_WINDOW}d"],
                order_recency + 1,
            )
            * has_order_history
    )
    derived[f"has_search_after_last_order_{RECENCY_WINDOW}d"] = (
            total[f"search_days_after_last_order_{RECENCY_WINDOW}d"].gt(0)
            & has_order_history.astype(bool)
    ).astype("int8")
    derived[f"has_cart_after_last_order_{RECENCY_WINDOW}d"] = (
            total[f"cart_days_after_last_order_{RECENCY_WINDOW}d"].gt(0)
            & has_order_history.astype(bool)
    ).astype("int8")

    for phase in ("1_7", "8_15", "16_23", "24_end"):
        derived[f"order_days_month_phase_{phase}_share_{RECENCY_WINDOW}d"] = (
            safe_ratio(
                total[f"order_days_month_phase_{phase}_{RECENCY_WINDOW}d"],
                order_frequency,
            )
        )
        derived[f"gmv_month_phase_{phase}_share_{RECENCY_WINDOW}d"] = (
            safe_ratio(
                total[f"gmv_month_phase_{phase}_{RECENCY_WINDOW}d"],
                total[f"gmv_{RECENCY_WINDOW}d"],
            )
        )

    derived[f"is_seen_{RECENCY_WINDOW}d"] = (
            1 - total[f"never_activity_{RECENCY_WINDOW}d"]
    )
    derived[f"btyd_frequency_{RECENCY_WINDOW}d"] = (
            order_frequency - 1
    ).clip(lower=0)
    derived[f"btyd_recency_{RECENCY_WINDOW}d"] = total[
        f"order_span_{RECENCY_WINDOW}d"
    ]
    derived[f"btyd_T_{RECENCY_WINDOW}d"] = order_age
    derived[f"order_frequency_per_age_{RECENCY_WINDOW}d"] = safe_ratio(
        order_frequency,
        order_age + 1,
    )
    derived[f"order_recency_to_age_{RECENCY_WINDOW}d"] = safe_ratio(
        order_recency,
        order_age + 1,
    )
    derived[f"order_frequency_per_recency_{RECENCY_WINDOW}d"] = safe_ratio(
        order_frequency,
        order_recency + 1,
    )

    derived["engaged_nonbuyer_30d"] = (
            total["active_days_30d"].gt(0)
            & total["order_days_30d"].eq(0)
    ).astype("int8")
    derived["recent_cart_without_order_30d"] = (
            total["cart_days_30d"].gt(0)
            & total["order_days_30d"].eq(0)
    ).astype("int8")
    derived["expected_gmv_from_recent_cart_30d"] = (
            total["items_added_to_cart_30d"]
            * safe_ratio(
        total["items_bought_180d"],
        total["items_added_to_cart_180d"],
    )
            * safe_ratio(
        total["gmv_180d"],
        total["items_bought_180d"],
    )
    )
    derived["search_intent_value_30d"] = (
            total["searches_30d"]
            * safe_ratio(
        total["search_items_bought_180d"],
        total["searches_180d"],
    )
            * safe_ratio(
        total["gmv_search_180d"],
        total["search_items_bought_180d"],
    )
    )

    add_block_statistics(total, derived)

    total = total.drop(columns=internal_columns)
    derived_frame = pd.DataFrame(derived, index=total.index)
    return pd.concat([total, derived_frame], axis=1, copy=False)


def reduce_memory(total):
    """Хранит признаки в float32, чтобы итоговые срезы не раздували RAM."""

    excluded = {"user_id", "cutoff_date", "target"}
    feature_columns = [
        column
        for column in total.columns
        if column not in excluded
    ]
    total[feature_columns] = (
        total[feature_columns]
        .replace([np.inf, -np.inf], 0)
        .astype("float32")
    )
    if "target" in total:
        total["target"] = total["target"].astype("float32")
    return total


def build_snapshot(cutoff, output_file, make_target=True):
    """Создаёт срез: одна строка на пользователя, признаки строго до cutoff."""

    cutoff = pd.Timestamp(cutoff)
    parquet = pq.ParquetFile(SOURCE)
    snapshot_parts = []

    for batch_number, complete_chunk in iter_complete_user_chunks(parquet):
        part = build_chunk_features(
            complete_chunk,
            cutoff,
            make_target,
        )
        snapshot_parts.append(part)
        print(f"Обработан батч {batch_number + 1}")

    total = pd.concat(
        snapshot_parts,
        ignore_index=True,
        copy=False,
    )
    total = total.sort_values("user_id").reset_index(drop=True)

    if total["user_id"].duplicated().any():
        raise ValueError("После обработки появились дубли user_id")

    total["cutoff_date"] = cutoff
    total = add_derived_features(total)
    total = reduce_memory(total)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_file
    total.to_parquet(output_path, index=False)
    print(
        f"Сохранено: {output_path}; "
        f"строк: {len(total)}; столбцов: {len(total.columns)}"
    )


if __name__ == "__main__":
    build_snapshot("2025-08-17", "train_aug_predict_from_2025-08-17.parquet")
    build_snapshot("2025-09-16", "train_sep_predict_from_2025-09-16.parquet")
    build_snapshot("2025-10-16", "train_oct_predict_from_2025-10-16.parquet")
    build_snapshot("2025-11-15", "train_nov_predict_from_2025-11-15.parquet")
    build_snapshot("2025-12-15", "train_dec_predict_from_2025-12-15.parquet")
    build_snapshot("2026-01-14", "train_jan_predict_from_2026-01-14.parquet")
    build_snapshot("2026-02-13", "competition_test.parquet", make_target=False)
