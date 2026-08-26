import os

import numpy as np
import pandas as pd


GAMMA = 1.1

all_regressor_file = (
    "C:/Users/myska/Downloads/Telegram Desktop/catboost_user_errors (1).csv"
)
positive_regressor_file = "C:/Users/myska/Downloads/catboost_user_errors_1.csv"
classifier_file = (
    "C:/Users/myska/OneDrive/Рабочий стол/catboost_classifier_fold_predictions.csv"
)

output_dir = "result_on_fold"
output_file = "result_on_fold/cnn_residual_target_hurdle_gamma_1_1.csv"
summary_file = "result_on_fold/cnn_residual_target_hurdle_gamma_1_1_summary.csv"


os.makedirs(output_dir, exist_ok=True)

all_regressor = pd.read_csv(all_regressor_file)
positive_regressor = pd.read_csv(positive_regressor_file)
classifier = pd.read_csv(classifier_file)


all_regressor = all_regressor.rename(
    columns={
        "fold": "fold_all_regressor",
        "target_true": "target_all_regressor",
        "pred_expm1": "predict_all_regressor",
    }
)

positive_regressor = positive_regressor.rename(
    columns={
        "fold": "fold_positive_regressor",
        "target_true": "target_positive_regressor",
        "pred_expm1": "predict_positive_regressor",
    }
)

classifier = classifier.rename(
    columns={
        "fold": "fold_classifier",
        "target_true": "target_classifier",
        "pred_positive_probability": "positive_probability",
    }
)


join_columns = ["user_id", "snapshot_name"]

data = positive_regressor[
    join_columns
    + [
        "fold_positive_regressor",
        "target_positive_regressor",
        "predict_positive_regressor",
    ]
].merge(
    classifier[
        join_columns
        + [
            "fold_classifier",
            "target_classifier",
            "target_positive_true",
            "positive_probability",
        ]
    ],
    on=join_columns,
    how="inner",
)

data = data.merge(
    all_regressor[
        join_columns
        + [
            "fold_all_regressor",
            "target_all_regressor",
            "predict_all_regressor",
        ]
    ],
    on=join_columns,
    how="inner",
)


target_difference_1 = (
    data["target_positive_regressor"] - data["target_classifier"]
).abs().max()
target_difference_2 = (
    data["target_positive_regressor"] - data["target_all_regressor"]
).abs().max()

if target_difference_1 > 0.000001 or target_difference_2 > 0.000001:
    raise ValueError("target не совпал между файлами")

if data.duplicated(join_columns).sum() > 0:
    raise ValueError("есть дубли по user_id + snapshot_name")


data["target"] = data["target_positive_regressor"]
data["target_log"] = np.log1p(data["target"])

data["predict_positive_log"] = np.log1p(data["predict_positive_regressor"])
data["predict_all_log"] = np.log1p(data["predict_all_regressor"])

data["positive_probability"] = data["positive_probability"].clip(0, 1)
data["probability_weight"] = data["positive_probability"] ** GAMMA

data["base_hurdle_log"] = (
    data["probability_weight"] * data["predict_positive_log"]
)
data["base_hurdle_predict"] = np.expm1(data["base_hurdle_log"])

data["cnn_target_residual"] = data["target_log"] - data["base_hurdle_log"]
data["cnn_target_residual_abs"] = data["cnn_target_residual"].abs()

clip_low = data["cnn_target_residual"].quantile(0.005)
clip_high = data["cnn_target_residual"].quantile(0.995)
data["cnn_target_residual_clipped"] = data["cnn_target_residual"].clip(
    clip_low,
    clip_high,
)


result = data[
    [
        "user_id",
        "snapshot_name",
        "fold_classifier",
        "fold_positive_regressor",
        "fold_all_regressor",
        "target",
        "target_positive_true",
        "target_log",
        "positive_probability",
        "probability_weight",
        "predict_positive_regressor",
        "predict_positive_log",
        "base_hurdle_predict",
        "base_hurdle_log",
        "cnn_target_residual",
        "cnn_target_residual_abs",
        "cnn_target_residual_clipped",
        "predict_all_regressor",
        "predict_all_log",
    ]
]

result = result.sort_values(["snapshot_name", "user_id"]).reset_index(drop=True)
result.to_csv(output_file, index=False)


all_residual = result["target_log"] - result["predict_all_log"]
positive_residual = result["target_log"] - result["predict_positive_log"]
hurdle_residual = result["cnn_target_residual"]

summary = pd.DataFrame(
    {
        "metric": [
            "rows",
            "unique_user_snapshot",
            "zero_target_share",
            "all_regressor_rmsle",
            "positive_regressor_rmsle",
            "hurdle_gamma_1_1_rmsle",
            "residual_clip_low",
            "residual_clip_high",
        ],
        "value": [
            len(result),
            result.drop_duplicates(join_columns).shape[0],
            (result["target"] == 0).mean(),
            np.sqrt(np.mean(all_residual**2)),
            np.sqrt(np.mean(positive_residual**2)),
            np.sqrt(np.mean(hurdle_residual**2)),
            clip_low,
            clip_high,
        ],
    }
)

summary.to_csv(summary_file, index=False)

print(f"saved {output_file}")
print(f"saved {summary_file}")
print(summary)
