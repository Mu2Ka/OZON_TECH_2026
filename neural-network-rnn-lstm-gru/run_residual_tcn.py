import gc

import numpy as np
import pandas as pd
import torch

from residual_tcn_dataset import get_feature_columns
from residual_tcn_model import ResidualTCNModel
from residual_tcn_training import (
    find_best_alpha,
    fit_tcn_on_all_train,
    predict_residual,
    predict_residual_validation,
    train_tcn_with_validation,
)


GAMMA = 1.1

RESIDUAL_FILE = "/kaggle/working/cnn_residual_target_hurdle_gamma_1_1.csv"

SEQUENCE_NOV = (
    "/kaggle/input/datasets/mu2kagg/sequenceozontech/"
    "sequence_cutoff2025-11-15_features.parquet"
)
SEQUENCE_DEC = (
    "/kaggle/input/datasets/mu2kagg/sequenceozontech/"
    "sequence_cutoff2025-12-15_features.parquet"
)
SEQUENCE_JAN = (
    "/kaggle/input/datasets/mu2kagg/sequenceozontech/"
    "sequence_cutoff2026-01-14_features.parquet"
)
SEQUENCE_TEST = (
    "/kaggle/input/datasets/mu2kagg/sequenceozontech/"
    "sequence_cutoff2026-02-13_features.parquet"
)

CLASSIFIER_TEST_FILE = "/kaggle/working/catboost_classifier_test_probability.csv"
POSITIVE_REGRESSOR_TEST_FILE = (
    "/kaggle/working/catboost_competition_test_predictions_1.csv"
)

TARGET_COLUMN = "cnn_target_residual_clipped"

HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.001


def get_prediction_column(data):
    if "predict" in data.columns:
        return "predict"
    if "pred_expm1" in data.columns:
        return "pred_expm1"
    if "pred_positive_probability" in data.columns:
        return "pred_positive_probability"

    raise ValueError(f"Не нашел колонку с прогнозом: {data.columns.tolist()}")


def make_submission(
        residual_prediction,
        classifier_test_file,
        positive_regressor_test_file,
        alpha,
        output_file,
):
    classifier = pd.read_csv(classifier_test_file)
    positive_regressor = pd.read_csv(positive_regressor_test_file)

    classifier_predict_column = get_prediction_column(classifier)
    positive_predict_column = get_prediction_column(positive_regressor)

    classifier = classifier[["user_id", classifier_predict_column]].rename(
        columns={classifier_predict_column: "positive_probability"}
    )
    positive_regressor = positive_regressor[
        ["user_id", positive_predict_column]
    ].rename(
        columns={positive_predict_column: "predict_positive_regressor"}
    )

    submission = classifier.merge(
        positive_regressor,
        on="user_id",
        how="inner",
    )
    submission = submission.merge(
        residual_prediction,
        on="user_id",
        how="inner",
    )

    submission["positive_probability"] = (
        submission["positive_probability"].clip(0, 1)
    )
    submission["base_hurdle_log"] = (
        (submission["positive_probability"] ** GAMMA)
        * np.log1p(submission["predict_positive_regressor"].clip(lower=0))
    )
    submission["final_log"] = (
        submission["base_hurdle_log"]
        + alpha * submission["tcn_residual_prediction"]
    )
    submission["final_log"] = submission["final_log"].clip(lower=0)
    submission["predict"] = np.expm1(submission["final_log"])

    submission[["user_id", "predict"]].to_csv(output_file, index=False)
    print(f"Saved submission: {output_file}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_columns = get_feature_columns(SEQUENCE_NOV)
input_size = len(feature_columns)

folds = [
    {
        "train_files": [SEQUENCE_NOV],
        "valid_file": SEQUENCE_DEC,
    },
    {
        "train_files": [SEQUENCE_NOV, SEQUENCE_DEC],
        "valid_file": SEQUENCE_JAN,
    },
]

best_epochs = []
validation_predictions = []

for fold_number, fold in enumerate(folds, start=1):
    print(f"Fold {fold_number}")

    model = ResidualTCNModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        dropout=0.1,
    )

    best_epoch, model, train_losses, valid_losses = train_tcn_with_validation(
        model=model,
        train_files=fold["train_files"],
        valid_file=fold["valid_file"],
        residual_file=RESIDUAL_FILE,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        epochs=EPOCHS,
        device=device,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    best_epochs.append(best_epoch)

    fold_prediction = predict_residual_validation(
        model=model,
        sequence_file=fold["valid_file"],
        residual_file=RESIDUAL_FILE,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        device=device,
        batch_size=BATCH_SIZE,
    )
    fold_prediction["fold"] = fold_number
    validation_predictions.append(fold_prediction)

    fold_prediction.to_csv(
        f"/kaggle/working/tcn_residual_valid_fold_{fold_number}.csv",
        index=False,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


validation_predictions = pd.concat(
    validation_predictions,
    ignore_index=True,
)
validation_predictions.to_csv(
    "/kaggle/working/tcn_residual_oof_predictions.csv",
    index=False,
)

alphas = [
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.25,
    0.30,
]
alpha_result = find_best_alpha(
    validation_prediction=validation_predictions,
    alphas=alphas,
)
alpha_result.to_csv("/kaggle/working/tcn_alpha_validation.csv", index=False)
print(alpha_result)

best_alpha = float(alpha_result.iloc[0]["alpha"])
final_epochs = int(round(np.mean(best_epochs)))
final_epochs = max(final_epochs, 1)

print(f"best_alpha={best_alpha}")
print(f"final_epochs={final_epochs}")

final_model = ResidualTCNModel(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    dropout=0.1,
)
final_model, final_train_losses = fit_tcn_on_all_train(
    model=final_model,
    train_files=[SEQUENCE_NOV, SEQUENCE_DEC, SEQUENCE_JAN],
    residual_file=RESIDUAL_FILE,
    feature_columns=feature_columns,
    target_column=TARGET_COLUMN,
    epochs=final_epochs,
    device=device,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
)

test_residual_prediction = predict_residual(
    model=final_model,
    sequence_file=SEQUENCE_TEST,
    feature_columns=feature_columns,
    device=device,
    batch_size=BATCH_SIZE,
)
test_residual_prediction.to_csv(
    "/kaggle/working/tcn_residual_test_prediction.csv",
    index=False,
)

make_submission(
    residual_prediction=test_residual_prediction,
    classifier_test_file=CLASSIFIER_TEST_FILE,
    positive_regressor_test_file=POSITIVE_REGRESSOR_TEST_FILE,
    alpha=best_alpha,
    output_file="/kaggle/working/submission_hurdle_tcn_residual.csv",
)
