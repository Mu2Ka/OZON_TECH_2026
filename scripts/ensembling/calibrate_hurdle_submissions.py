import os

import numpy as np
import pandas as pd


classifier_oof_file = (
    "C:/Users/myska/OneDrive/Рабочий стол/"
    "catboost_classifier_fold_predictions.csv"
)
positive_regressor_oof_file = (
    "C:/Users/myska/Downloads/catboost_user_errors_1.csv"
)

classifier_test_file = (
    "C:/Users/myska/OneDrive/Рабочий стол/"
    "catboost_classifier_test_probability.csv"
)
positive_regressor_test_file = (
    "C:/Users/myska/Downloads/Telegram Desktop/"
    "catboost_competition_test_predictions_1.csv"
)

output_dir = "result_on_fold"
summary_file = "result_on_fold/hurdle_calibration_summary.csv"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 0.000001, 0.999999)
    return np.log(p / (1.0 - p))


def rmsle_from_log(target_log, predict_log):
    return float(np.sqrt(np.mean((target_log - predict_log) ** 2)))


def logloss(y, p):
    p = np.clip(p, 0.000001, 0.999999)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def make_log_prediction(probability, positive_log, gamma, scale):
    prediction_log = (probability ** gamma) * (scale * positive_log)
    prediction_log = np.maximum(prediction_log, 0)
    return prediction_log


def score_arrays(target_log, probability, positive_log, gamma, scale):
    prediction_log = make_log_prediction(
        probability=probability,
        positive_log=positive_log,
        gamma=gamma,
        scale=scale,
    )

    return rmsle_from_log(
        target_log=target_log,
        predict_log=prediction_log,
    )


def make_submission(
        test,
        probability_column,
        gamma,
        scale,
        output_file,
):
    prediction_log = make_log_prediction(
        probability=test[probability_column].to_numpy(),
        positive_log=test["positive_log"].to_numpy(),
        gamma=gamma,
        scale=scale,
    )
    prediction = np.expm1(prediction_log)

    submission = pd.DataFrame(
        {
            "user_id": test["user_id"],
            "predict": prediction,
        }
    )
    submission.to_csv(output_file, index=False)
    return submission


def output_name(name):
    return f"result_on_fold/{name}.csv"


os.makedirs(output_dir, exist_ok=True)

classifier = pd.read_csv(classifier_oof_file)
positive_regressor = pd.read_csv(positive_regressor_oof_file)

classifier = classifier.rename(
    columns={
        "pred_positive_probability": "probability_raw",
        "target_true": "target_classifier",
    }
)
positive_regressor = positive_regressor.rename(
    columns={
        "pred_expm1": "positive_prediction",
        "target_true": "target_regressor",
    }
)

classifier["target_positive_true"] = classifier["target_positive_true"].astype(int)
classifier["probability_raw"] = classifier["probability_raw"].clip(0.000001, 0.999999)


best_logloss = 10**9
best_platt_a = 1.0
best_platt_b = 0.0

classifier_logit = logit(classifier["probability_raw"].to_numpy())
classifier_target = classifier["target_positive_true"].to_numpy()

for a in np.round(np.arange(0.80, 1.201, 0.05), 3):
    for b in np.round(np.arange(-0.15, 0.151, 0.05), 3):
        probability = sigmoid(a * classifier_logit + b)
        current_logloss = logloss(classifier_target, probability)

        if current_logloss < best_logloss:
            best_logloss = current_logloss
            best_platt_a = float(a)
            best_platt_b = float(b)

classifier["probability_platt"] = sigmoid(
    best_platt_a * classifier_logit + best_platt_b
)


join_columns = ["user_id", "snapshot_name"]

data = positive_regressor[
    join_columns + ["target_regressor", "positive_prediction"]
].merge(
    classifier[
        join_columns
        + [
            "target_classifier",
            "target_positive_true",
            "probability_raw",
            "probability_platt",
        ]
    ],
    on=join_columns,
    how="inner",
)

target_difference = (
    data["target_regressor"] - data["target_classifier"]
).abs().max()
if target_difference > 0.000001:
    raise ValueError("target не совпал между classifier и positive regressor")

if data.duplicated(join_columns).sum() > 0:
    raise ValueError("Есть дубли по user_id + snapshot_name")

data["target"] = data["target_regressor"]
data["target_log"] = np.log1p(data["target"])
data["positive_prediction"] = data["positive_prediction"].clip(lower=0)
data["positive_log"] = np.log1p(data["positive_prediction"])


rows = []
gammas = np.round(np.arange(1.00, 1.151, 0.01), 3)
scales = np.round(np.arange(0.96, 1.041, 0.01), 3)
probability_columns = ["probability_raw", "probability_platt"]

target_log_array = data["target_log"].to_numpy()
positive_log_array = data["positive_log"].to_numpy()
snapshot_names = data["snapshot_name"].to_numpy()
snapshot_unique = sorted(data["snapshot_name"].unique())

for probability_column in probability_columns:
    probability_array = data[probability_column].to_numpy()

    for gamma in gammas:
        for scale in scales:
            rmsle = score_arrays(
                target_log=target_log_array,
                probability=probability_array,
                positive_log=positive_log_array,
                gamma=gamma,
                scale=scale,
            )

            row = {
                "probability": probability_column,
                "gamma": gamma,
                "scale": scale,
                "rmsle": rmsle,
            }
            rows.append(row)

summary = pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)

top_for_snapshot = summary.head(120).copy()

snapshot_rows = []
for row in top_for_snapshot.to_dict("records"):
    probability_array = data[row["probability"]].to_numpy()
    output_row = row.copy()

    for snapshot_name in snapshot_unique:
        mask = snapshot_names == snapshot_name
        output_row[f"rmsle_{snapshot_name}"] = score_arrays(
            target_log=target_log_array[mask],
            probability=probability_array[mask],
            positive_log=positive_log_array[mask],
            gamma=row["gamma"],
            scale=row["scale"],
        )

    snapshot_rows.append(output_row)

summary = pd.DataFrame(snapshot_rows)
snapshot_columns = [
    column for column in summary.columns if column.startswith("rmsle_train_")
]
summary["worst_snapshot_rmsle"] = summary[snapshot_columns].max(axis=1)
summary["std_snapshot_rmsle"] = summary[snapshot_columns].std(axis=1)
summary["stable_score"] = (
    summary["rmsle"] + 0.25 * summary["std_snapshot_rmsle"]
)

summary = summary.sort_values("rmsle").reset_index(drop=True)
summary.to_csv(summary_file, index=False)


classifier_test = pd.read_csv(classifier_test_file)
positive_test = pd.read_csv(positive_regressor_test_file)

classifier_test = classifier_test.rename(columns={"predict": "probability_raw"})
positive_test = positive_test.rename(columns={"pred_expm1": "positive_prediction"})

test = classifier_test[["user_id", "probability_raw"]].merge(
    positive_test[["user_id", "positive_prediction"]],
    on="user_id",
    how="inner",
)

if len(test) != len(classifier_test):
    raise ValueError("После merge test потерялись пользователи")

test["probability_raw"] = test["probability_raw"].clip(0.000001, 0.999999)
test["probability_platt"] = sigmoid(
    best_platt_a * logit(test["probability_raw"].to_numpy()) + best_platt_b
)
test["positive_prediction"] = test["positive_prediction"].clip(lower=0)
test["positive_log"] = np.log1p(test["positive_prediction"])


best_global = summary.iloc[0]
best_stable = summary.sort_values("stable_score").iloc[0]
best_jan = summary.sort_values("rmsle_train_jan_predict_from_2026-01-14").iloc[0]

baseline_gamma_1_1 = summary.loc[
    (summary["probability"].eq("probability_raw"))
    & (summary["gamma"].eq(1.1))
    & (summary["scale"].eq(1.0))
].iloc[0]

candidates = [
    ("calibrated_best_global_oof", best_global),
    ("calibrated_best_stable_oof", best_stable),
    ("calibrated_best_jan_oof", best_jan),
    ("calibrated_baseline_gamma_1_1", baseline_gamma_1_1),
]

created_files = []
for name, row in candidates:
    file_name = output_name(f"submission_{name}")
    make_submission(
        test=test,
        probability_column=row["probability"],
        gamma=float(row["gamma"]),
        scale=float(row["scale"]),
        output_file=file_name,
    )
    created_files.append(file_name)


extra_small_grid = [
    ("submission_raw_gamma_1_095_scale_1_000", "probability_raw", 1.095, 1.0),
    ("submission_raw_gamma_1_100_scale_1_005", "probability_raw", 1.1, 1.005),
    ("submission_raw_gamma_1_105_scale_1_000", "probability_raw", 1.105, 1.0),
    ("submission_raw_gamma_1_100_scale_0_995", "probability_raw", 1.1, 0.995),
]

for name, probability_column, gamma, scale in extra_small_grid:
    file_name = output_name(name)
    make_submission(
        test=test,
        probability_column=probability_column,
        gamma=gamma,
        scale=scale,
        output_file=file_name,
    )
    created_files.append(file_name)


print("summary", summary_file)
print("best_platt_a", best_platt_a)
print("best_platt_b", best_platt_b)
print("best_logloss", best_logloss)
print("baseline gamma=1.1")
print(baseline_gamma_1_1.to_string())
print("best global")
print(best_global.to_string())
print("best stable")
print(best_stable.to_string())
print("best jan")
print(best_jan.to_string())
print("created files")
for file_name in created_files:
    print(file_name)
