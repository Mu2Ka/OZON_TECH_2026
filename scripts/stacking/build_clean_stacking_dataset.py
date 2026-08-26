from pathlib import Path
import shutil

import numpy as np
import pandas as pd


DESKTOP = Path.home() / "OneDrive" / "\u0420\u0430\u0431\u043e\u0447\u0438\u0439 \u0441\u0442\u043e\u043b"

CAT_HURDLE_DIR = DESKTOP / "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b \u043c\u043e\u0434\u0435\u043b\u0438" / "classifier+regression_on_pos+gamma 1.1"
CAT_DIRECT_DIR = DESKTOP / "\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b \u043c\u043e\u0434\u0435\u043b\u0438" / "\u0441atboost regressor \u043d\u0430 \u0432\u0441\u0435\u0445"
XGB_FAST_DIR = DESKTOP / "result xgb fast"
XGB_ALL_DIR = DESKTOP / "xgboost all month"
LGBM_DIR = DESKTOP / "lgb,"

OUT_DIR = Path("stacking_dataset_clean")
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(exist_ok=True)


def read_csv(path, columns=None):
    if columns is None:
        return pd.read_csv(path)
    return pd.read_csv(path, usecols=columns)


def ensure_unique(df, keys, name):
    duplicates = int(df.duplicated(keys).sum())
    if duplicates:
        raise ValueError(f"{name}: {duplicates} duplicate rows by {keys}")


def add_log_from_pred(df, pred_col, log_col):
    df[log_col] = np.log1p(df[pred_col].clip(lower=0))
    return df


def read_oof_regression(path, name):
    df = read_csv(path, ["user_id", "snapshot_name", "target_true", "pred_log", "pred_expm1"])
    df = df.rename(
        columns={
            "target_true": "target",
            "pred_log": f"{name}_log",
            "pred_expm1": f"{name}_pred",
        }
    )
    ensure_unique(df, ["user_id", "snapshot_name"], name)
    return df


def read_oof_classifier(path, name):
    df = read_csv(path, ["user_id", "snapshot_name", "target_true", "pred_positive_probability"])
    df = df.rename(
        columns={
            "target_true": "target",
            "pred_positive_probability": f"{name}_p",
        }
    )
    ensure_unique(df, ["user_id", "snapshot_name"], name)
    return df


def read_fast_oof(path, snapshot_name, name):
    df = read_csv(path, ["user_id", "target", "predict", "predict_log"])
    df["snapshot_name"] = snapshot_name
    df = df.rename(
        columns={
            "predict": f"{name}_pred",
            "predict_log": f"{name}_log",
        }
    )
    df = df[["user_id", "snapshot_name", "target", f"{name}_log", f"{name}_pred"]]
    ensure_unique(df, ["user_id", "snapshot_name"], name)
    return df


def merge_oof(base, other, name):
    if "target" in other.columns:
        other_target = other[["user_id", "snapshot_name", "target"]]
        check = base[["user_id", "snapshot_name", "target"]].merge(
            other_target,
            on=["user_id", "snapshot_name"],
            how="inner",
            suffixes=("_base", f"_{name}"),
            validate="one_to_one",
        )
        bad = (check["target_base"] - check[f"target_{name}"]).abs().gt(1e-5).sum()
        if bad:
            raise ValueError(f"{name}: {bad} target mismatches")
        other = other.drop(columns=["target"])

    return base.merge(other, on=["user_id", "snapshot_name"], how="left", validate="one_to_one")


cat_direct = read_oof_regression(
    CAT_DIRECT_DIR / "catboost_fold_predictions.csv",
    "cat_direct",
)
cat_pos = read_oof_regression(
    CAT_HURDLE_DIR / "catboost_fold_predictions _1.csv",
    "cat_pos",
)
cat_cls = read_oof_classifier(
    CAT_HURDLE_DIR / "catboost_classifier_fold_predictions.csv",
    "cat_cls",
)
xgb_all = read_oof_regression(
    XGB_ALL_DIR / "xgb_all_months_oof.csv",
    "xgb_all",
)
lgbm_all = read_oof_regression(
    LGBM_DIR / "lgbm_all_months_oof.csv",
    "lgbm_all",
)
xgb_recent = pd.concat(
    [
        read_fast_oof(
            XGB_FAST_DIR / "xgb_fast_1_oct_to_nov.csv",
            "train_nov_predict_from_2025-11-15",
            "xgb_recent",
        ),
        read_fast_oof(
            XGB_FAST_DIR / "xgb_fast_2_nov_to_dec.csv",
            "train_dec_predict_from_2025-12-15",
            "xgb_recent",
        ),
        read_fast_oof(
            XGB_FAST_DIR / "xgb_fast_3_dec_to_jan.csv",
            "train_jan_predict_from_2026-01-14",
            "xgb_recent",
        ),
    ],
    ignore_index=True,
)

oof = cat_direct.copy()
for name, part in [
    ("cat_pos", cat_pos),
    ("cat_cls", cat_cls),
    ("xgb_all", xgb_all),
    ("lgbm_all", lgbm_all),
    ("xgb_recent", xgb_recent),
]:
    oof = merge_oof(oof, part, name)

oof["target_log"] = np.log1p(oof["target"].clip(lower=0))
oof["target_positive"] = oof["target"].gt(0).astype("int8")
oof["cat_hurdle_log_p1_0"] = (oof["cat_cls_p"].clip(0, 1) ** 1.0) * oof["cat_pos_log"]
oof["cat_hurdle_pred_p1_0"] = np.expm1(oof["cat_hurdle_log_p1_0"].clip(lower=0))
oof["cat_hurdle_log_p1_1"] = (oof["cat_cls_p"].clip(0, 1) ** 1.1) * oof["cat_pos_log"]
oof["cat_hurdle_pred_p1_1"] = np.expm1(oof["cat_hurdle_log_p1_1"].clip(lower=0))

first_cols = ["user_id", "snapshot_name", "target", "target_log", "target_positive"]
feature_cols = [column for column in oof.columns if column not in first_cols]
oof = oof[first_cols + sorted(feature_cols)]

oof_no_lgbm = (
    oof.drop(columns=["lgbm_all_log", "lgbm_all_pred"])
    .dropna(axis=0)
    .reset_index(drop=True)
)
oof_all_models = oof.dropna(axis=0).reset_index(drop=True)


def read_test_regression(path, name, has_log):
    df = read_csv(path)
    if has_log:
        df = df[["user_id", "pred_log", "pred_expm1"]].rename(
            columns={"pred_log": f"{name}_log", "pred_expm1": f"{name}_pred"}
        )
    else:
        df = df[["user_id", "predict"]].rename(columns={"predict": f"{name}_pred"})
        df = add_log_from_pred(df, f"{name}_pred", f"{name}_log")
        df = df[["user_id", f"{name}_log", f"{name}_pred"]]
    ensure_unique(df, ["user_id"], name)
    return df


def read_test_classifier(path, name):
    df = read_csv(path, ["user_id", "predict"])
    df = df.rename(columns={"predict": f"{name}_p"})
    ensure_unique(df, ["user_id"], name)
    return df


test = read_test_regression(
    CAT_DIRECT_DIR / "catboost_competition_test_predictions.csv",
    "cat_direct",
    has_log=True,
)
for name, part in [
    (
        "cat_pos",
        read_test_regression(
            CAT_HURDLE_DIR / "catboost_competition_test_predictions_1.csv",
            "cat_pos",
            has_log=True,
        ),
    ),
    (
        "cat_cls",
        read_test_classifier(
            CAT_HURDLE_DIR / "catboost_classifier_test_probability.csv",
            "cat_cls",
        ),
    ),
    (
        "xgb_all",
        read_test_regression(
            XGB_ALL_DIR / "submission_xgb_all_months.csv",
            "xgb_all",
            has_log=False,
        ),
    ),
    (
        "lgbm_all",
        read_test_regression(
            LGBM_DIR / "submission_lgbm_all_months.csv",
            "lgbm_all",
            has_log=False,
        ),
    ),
    (
        "xgb_recent",
        read_test_regression(
            XGB_FAST_DIR / "submission_xgb_last_2_months.csv",
            "xgb_recent",
            has_log=False,
        ),
    ),
]:
    test = test.merge(part, on="user_id", how="left", validate="one_to_one")

test["cat_hurdle_log_p1_0"] = (test["cat_cls_p"].clip(0, 1) ** 1.0) * test["cat_pos_log"]
test["cat_hurdle_pred_p1_0"] = np.expm1(test["cat_hurdle_log_p1_0"].clip(lower=0))
test["cat_hurdle_log_p1_1"] = (test["cat_cls_p"].clip(0, 1) ** 1.1) * test["cat_pos_log"]
test["cat_hurdle_pred_p1_1"] = np.expm1(test["cat_hurdle_log_p1_1"].clip(lower=0))

test_feature_cols = [column for column in test.columns if column != "user_id"]
test_all_models = test[["user_id"] + sorted(test_feature_cols)].copy()
test_no_lgbm = test_all_models.drop(columns=["lgbm_all_log", "lgbm_all_pred"]).copy()

for df in [oof_no_lgbm, oof_all_models, test_no_lgbm, test_all_models]:
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

oof_no_lgbm_path = OUT_DIR / "stack_train_no_lgbm_750k.csv"
test_no_lgbm_path = OUT_DIR / "stack_test_no_lgbm.csv"
oof_all_models_path = OUT_DIR / "stack_train_all_models_500k.csv"
test_all_models_path = OUT_DIR / "stack_test_all_models.csv"

oof_no_lgbm.to_csv(oof_no_lgbm_path, index=False)
test_no_lgbm.to_csv(test_no_lgbm_path, index=False)
oof_all_models.to_csv(oof_all_models_path, index=False)
test_all_models.to_csv(test_all_models_path, index=False)

try:
    oof_no_lgbm.to_parquet(OUT_DIR / "stack_train_no_lgbm_750k.parquet", index=False)
    test_no_lgbm.to_parquet(OUT_DIR / "stack_test_no_lgbm.parquet", index=False)
    oof_all_models.to_parquet(OUT_DIR / "stack_train_all_models_500k.parquet", index=False)
    test_all_models.to_parquet(OUT_DIR / "stack_test_all_models.parquet", index=False)
except Exception as exc:
    print("Parquet skipped:", exc)

archive_path = shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)

print("Saved:")
print(oof_no_lgbm_path, oof_no_lgbm.shape)
print(test_no_lgbm_path, test_no_lgbm.shape)
print(oof_all_models_path, oof_all_models.shape)
print(test_all_models_path, test_all_models.shape)
print(archive_path)
print("Columns:")
print("no_lgbm:", list(oof_no_lgbm.columns))
print("all_models:", list(oof_all_models.columns))
