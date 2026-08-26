import os

import numpy as np
import pandas as pd


GAMMA = 1.1
MIN_TRUSTED_GAIN = 0.001

valid_fold_1_file = "C:/Users/myska/OneDrive/Рабочий стол/tcn_residual_valid_fold_1.csv"
valid_fold_2_file = "C:/Users/myska/OneDrive/Рабочий стол/tcn_residual_valid_fold_2.csv"
test_residual_file = "C:/Users/myska/OneDrive/Рабочий стол/tcn_residual_test_prediction.csv"

classifier_test_file = (
    "C:/Users/myska/OneDrive/Рабочий стол/catboost_classifier_test_probability.csv"
)
positive_regressor_test_file = (
    "C:/Users/myska/Downloads/Telegram Desktop/"
    "catboost_competition_test_predictions_1.csv"
)

output_dir = "result_on_fold"
alpha_table_file = "result_on_fold/tcn_alpha_validation_from_uploaded_files.csv"
final_submission_file = "result_on_fold/submission_final_hurdle_tcn_selected.csv"
best_oof_submission_file = "result_on_fold/submission_hurdle_tcn_best_oof_alpha.csv"
small_alpha_submission_file = "result_on_fold/submission_hurdle_tcn_alpha_0_05.csv"
baseline_submission_file = "result_on_fold/submission_hurdle_gamma_1_1_alpha_0.csv"


def rmsle_from_logs(true_log, predict_log):
    error = true_log - predict_log
    return float(np.sqrt(np.mean(error**2)))


def score_alpha(validation, alpha):
    final_log = validation["base_hurdle_log"] + alpha * validation[
        "tcn_residual_prediction"
    ]
    final_log = np.maximum(final_log, 0)

    return rmsle_from_logs(validation["target_log"], final_log)


def get_predict_column(data):
    if "predict" in data.columns:
        return "predict"
    if "pred_expm1" in data.columns:
        return "pred_expm1"

    raise ValueError(f"Не нашел колонку прогноза: {data.columns.tolist()}")


def make_submission(test_residual, alpha, output_file):
    classifier = pd.read_csv(classifier_test_file)
    positive_regressor = pd.read_csv(positive_regressor_test_file)

    classifier_predict_column = get_predict_column(classifier)
    positive_predict_column = get_predict_column(positive_regressor)

    classifier = classifier[["user_id", classifier_predict_column]].rename(
        columns={classifier_predict_column: "positive_probability"}
    )
    positive_regressor = positive_regressor[
        ["user_id", positive_predict_column]
    ].rename(columns={positive_predict_column: "positive_regressor_predict"})

    submission = classifier.merge(
        positive_regressor,
        on="user_id",
        how="inner",
    )
    submission = submission.merge(
        test_residual[["user_id", "tcn_residual_prediction"]],
        on="user_id",
        how="inner",
    )

    if len(submission) != len(classifier):
        raise ValueError("После merge потерялись пользователи")

    submission["positive_probability"] = submission["positive_probability"].clip(0, 1)
    submission["positive_regressor_predict"] = submission[
        "positive_regressor_predict"
    ].clip(lower=0)

    submission["base_hurdle_log"] = (
        submission["positive_probability"] ** GAMMA
    ) * np.log1p(submission["positive_regressor_predict"])

    submission["final_log"] = (
        submission["base_hurdle_log"]
        + alpha * submission["tcn_residual_prediction"]
    )
    submission["final_log"] = submission["final_log"].clip(lower=0)
    submission["predict"] = np.expm1(submission["final_log"])

    submission[["user_id", "predict"]].to_csv(output_file, index=False)

    return submission


os.makedirs(output_dir, exist_ok=True)

valid_fold_1 = pd.read_csv(valid_fold_1_file)
valid_fold_2 = pd.read_csv(valid_fold_2_file)
validation = pd.concat([valid_fold_1, valid_fold_2], ignore_index=True)

alphas = np.round(np.arange(-0.30, 0.301, 0.005), 3)

rows = []
for alpha in alphas:
    fold_1_rmsle = score_alpha(valid_fold_1, alpha)
    fold_2_rmsle = score_alpha(valid_fold_2, alpha)
    all_rmsle = score_alpha(validation, alpha)

    rows.append(
        {
            "alpha": alpha,
            "rmsle": all_rmsle,
            "fold_1_rmsle": fold_1_rmsle,
            "fold_2_rmsle": fold_2_rmsle,
        }
    )

alpha_table = pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)
alpha_table.to_csv(alpha_table_file, index=False)

base_rmsle = float(alpha_table.loc[alpha_table["alpha"].eq(0), "rmsle"].iloc[0])
best_oof_alpha = float(alpha_table.iloc[0]["alpha"])
best_oof_rmsle = float(alpha_table.iloc[0]["rmsle"])
oof_gain = base_rmsle - best_oof_rmsle

if oof_gain >= MIN_TRUSTED_GAIN:
    selected_alpha = best_oof_alpha
else:
    selected_alpha = 0.0

test_residual = pd.read_csv(test_residual_file)

final_submission = make_submission(
    test_residual=test_residual,
    alpha=selected_alpha,
    output_file=final_submission_file,
)
best_oof_submission = make_submission(
    test_residual=test_residual,
    alpha=best_oof_alpha,
    output_file=best_oof_submission_file,
)
small_alpha_submission = make_submission(
    test_residual=test_residual,
    alpha=0.05,
    output_file=small_alpha_submission_file,
)
baseline_submission = make_submission(
    test_residual=test_residual,
    alpha=0.0,
    output_file=baseline_submission_file,
)

print("alpha table:", alpha_table_file)
print(alpha_table.head(20).to_string(index=False))
print("base_rmsle", base_rmsle)
print("best_oof_alpha", best_oof_alpha)
print("best_oof_rmsle", best_oof_rmsle)
print("oof_gain", oof_gain)
print("selected_alpha", selected_alpha)
print("final submission:", final_submission_file)
print("best oof alpha submission:", best_oof_submission_file)
print("small alpha submission:", small_alpha_submission_file)
print("baseline submission:", baseline_submission_file)
print("final rows", len(final_submission))
print("final predict min", final_submission["predict"].min())
print("final predict max", final_submission["predict"].max())
print("baseline equals final", final_submission.equals(baseline_submission))
