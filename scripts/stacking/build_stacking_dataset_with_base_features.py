import os
import numpy as np
import pandas as pd


BASE_DIR = "/kaggle/input/datasets/mu2kagg/ozontech"
STACK_DIR = "/kaggle/input/stacking-dataset-clean"
OUT_DIR = "/kaggle/working/stacking_dataset_with_base_features"
os.makedirs(OUT_DIR, exist_ok=True)


TRAIN_NOV = f"{BASE_DIR}/train_nov_predict_from_2025-11-15.parquet"
TRAIN_DEC = f"{BASE_DIR}/train_dec_predict_from_2025-12-15.parquet"
TRAIN_JAN = f"{BASE_DIR}/train_jan_predict_from_2026-01-14.parquet"
TEST = f"{BASE_DIR}/competition_test.parquet"

STACK_TRAIN = f"{STACK_DIR}/stack_train_no_lgbm_750k.csv"
STACK_TEST = f"{STACK_DIR}/stack_test_no_lgbm.csv"


META_FEATURES = [
    "cat_cls_p",
    "cat_direct_log",
    "cat_direct_pred",
    "cat_hurdle_log_p1_0",
    "cat_hurdle_log_p1_1",
    "cat_hurdle_pred_p1_0",
    "cat_hurdle_pred_p1_1",
    "cat_pos_log",
    "cat_pos_pred",
    "xgb_all_log",
    "xgb_all_pred",
    "xgb_recent_log",
    "xgb_recent_pred",
]


BASE_FEATURE_CANDIDATES = [
    "gmv_7d", "gmv_14d", "gmv_30d", "gmv_60d", "gmv_90d", "gmv_180d", "gmv_270d", "gmv_365d",
    "orders_7d", "orders_14d", "orders_30d", "orders_60d", "orders_90d", "orders_180d", "orders_270d",
    "order_days_7d", "order_days_14d", "order_days_30d", "order_days_60d", "order_days_90d", "order_days_180d",
    "searches_7d", "searches_14d", "searches_30d", "searches_60d", "searches_90d",
    "carts_7d", "carts_14d", "carts_30d", "carts_60d", "carts_90d",
    "days_since_last_order", "days_since_last_search", "days_since_last_cart",
    "active_days_7d", "active_days_14d", "active_days_30d", "active_days_60d", "active_days_90d",
    "gmv_per_order_30d", "gmv_per_order_day_30d",
    "avg_order_value_30d", "avg_order_value_90d",
    "search_to_cart_30d", "cart_to_order_30d", "search_to_order_30d",
    "zero_order_streak", "current_zero_order_streak",
    "purchase_episodes_90d", "purchase_episodes_180d",
    "mean_days_between_orders", "std_days_between_orders",
]


BASE_FEATURE_KEYWORDS = [
    "gmv_7", "gmv_14", "gmv_30", "gmv_60", "gmv_90", "gmv_180", "gmv_270", "gmv_365",
    "order_7", "order_14", "order_30", "order_60", "order_90", "order_180", "order_270",
    "search_7", "search_14", "search_30", "search_60", "search_90",
    "cart_7", "cart_14", "cart_30", "cart_60", "cart_90",
    "recency", "days_since", "last_order", "last_search", "last_cart",
    "active_days", "order_days",
    "ratio_7_30", "ratio_14_30", "ratio_30_90", "ratio_90_270",
    "gmv_per", "avg_order", "aov",
    "funnel", "search_to_cart", "cart_to_order", "search_to_order",
    "episode", "interval", "streak",
]


def pick_base_features(parquet_file, max_features=45):
    columns = pd.read_parquet(parquet_file).columns.tolist()
    excluded = {"user_id", "cutoff_date", "target"}

    selected = []

    for col in BASE_FEATURE_CANDIDATES:
        if col in columns and col not in excluded and col not in selected:
            selected.append(col)

    for keyword in BASE_FEATURE_KEYWORDS:
        for col in columns:
            low = col.lower()
            if col in excluded or col in selected:
                continue
            if keyword in low:
                selected.append(col)
            if len(selected) >= max_features:
                return selected

    numeric_cols = []
    sample = pd.read_parquet(parquet_file, columns=[c for c in columns if c not in excluded][:120]).head(200)
    for col in sample.columns:
        if col not in selected and pd.api.types.is_numeric_dtype(sample[col]):
            numeric_cols.append(col)

    for col in numeric_cols:
        if col not in selected:
            selected.append(col)
        if len(selected) >= max_features:
            break

    return selected[:max_features]


BASE_FEATURES = pick_base_features(TRAIN_JAN, max_features=45)
print("BASE_FEATURES:", len(BASE_FEATURES))
print(BASE_FEATURES)


def add_base_features(stack_part, parquet_file):
    base = pd.read_parquet(parquet_file, columns=["user_id"] + BASE_FEATURES)
    base = base.rename(columns={col: f"base_{col}" for col in BASE_FEATURES})

    df = stack_part.merge(base, on="user_id", how="left", validate="one_to_one")

    base_cols = [f"base_{col}" for col in BASE_FEATURES]
    for col in base_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)

        if (
            "gmv" in col.lower()
            or "revenue" in col.lower()
            or "amount" in col.lower()
            or "price" in col.lower()
            or "value" in col.lower()
        ):
            df[col + "_log1p"] = np.log1p(df[col].clip(lower=0))

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


stack_train = pd.read_csv(STACK_TRAIN)
stack_test = pd.read_csv(STACK_TEST)


nov = stack_train[stack_train["snapshot_name"].eq("train_nov_predict_from_2025-11-15")].copy()
dec = stack_train[stack_train["snapshot_name"].eq("train_dec_predict_from_2025-12-15")].copy()
jan = stack_train[stack_train["snapshot_name"].eq("train_jan_predict_from_2026-01-14")].copy()
test = stack_test.copy()

nov = add_base_features(nov, TRAIN_NOV)
dec = add_base_features(dec, TRAIN_DEC)
jan = add_base_features(jan, TRAIN_JAN)
test = add_base_features(test, TEST)


TRAIN_FIRST_COLS = ["user_id", "snapshot_name", "target", "target_log", "target_positive"]
TEST_FIRST_COLS = ["user_id"]

train_feature_cols = sorted([c for c in jan.columns if c not in TRAIN_FIRST_COLS])
test_feature_cols = sorted([c for c in test.columns if c not in TEST_FIRST_COLS])

nov = nov[TRAIN_FIRST_COLS + train_feature_cols]
dec = dec[TRAIN_FIRST_COLS + train_feature_cols]
jan = jan[TRAIN_FIRST_COLS + train_feature_cols]
test = test[TEST_FIRST_COLS + test_feature_cols]

assert train_feature_cols == test_feature_cols
assert len(nov) == 250000
assert len(dec) == 250000
assert len(jan) == 250000
assert len(test) == 250000
assert nov.isna().sum().sum() == 0
assert dec.isna().sum().sum() == 0
assert jan.isna().sum().sum() == 0
assert test.isna().sum().sum() == 0


nov.to_csv(f"{OUT_DIR}/stack_train_nov_250k.csv", index=False)
dec.to_csv(f"{OUT_DIR}/stack_train_dec_250k.csv", index=False)
jan.to_csv(f"{OUT_DIR}/stack_train_jan_250k.csv", index=False)
test.to_csv(f"{OUT_DIR}/stack_test_competition_250k.csv", index=False)


all_train = pd.concat([nov, dec, jan], ignore_index=True)
all_train.to_csv(f"{OUT_DIR}/stack_train_all_750k.csv", index=False)

print("saved:")
print(f"{OUT_DIR}/stack_train_nov_250k.csv", nov.shape)
print(f"{OUT_DIR}/stack_train_dec_250k.csv", dec.shape)
print(f"{OUT_DIR}/stack_train_jan_250k.csv", jan.shape)
print(f"{OUT_DIR}/stack_test_competition_250k.csv", test.shape)
print(f"{OUT_DIR}/stack_train_all_750k.csv", all_train.shape)
print("feature_count:", len(train_feature_cols))
