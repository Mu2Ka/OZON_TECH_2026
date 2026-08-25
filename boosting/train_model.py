import gc
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_log_error
from sklearn.base import clone


def train_model_and_validation_catboost(
        model,
        train_files,
        validation_file,
        validation=True,
        selected_features=None,
        sample_weight=None,
        use_user_id=False,
        verbose=100,
        return_details=False,
        classifier=False,
        positive_only=False,
):
    if classifier and positive_only:
        raise ValueError("positive_only нельзя использовать вместе с classifier")

    if isinstance(train_files, dict):
        train_files_weight = train_files
        train_files = list(train_files_weight.keys())
    else:
        train_files_weight = None

    data_for_schema = pd.read_parquet(train_files[0])

    excluded_columns = ["user_id", "cutoff_date", "target"]

    if selected_features is None:
        features = [
            column
            for column in data_for_schema.columns
            if column not in excluded_columns
        ]
    else:
        features = selected_features

    if use_user_id:
        read_features = ["user_id"] + features
        model_features = features + ["user_id"]
        cat_features = ["user_id"]
    else:
        read_features = features
        model_features = features
        cat_features = None

    del data_for_schema
    gc.collect()

    if train_files_weight is None:
        train_data = pd.concat(
            (
                pd.read_parquet(
                    file,
                    columns=read_features + ["target"],
                )
                for file in train_files
            ),
            ignore_index=True,
            copy=False,
        )
    else:
        train_parts = []
        weight_parts = []

        for file, weight in train_files_weight.items():
            print(f"Загружаем {file}")

            data_part = pd.read_parquet(
                file,
                columns=read_features + ["target"],
            )

            train_parts.append(data_part)

            part_weights = np.full(
                len(data_part),
                weight,
                dtype=np.float32,
            )

            weight_parts.append(part_weights)

        train_data = pd.concat(
            train_parts,
            ignore_index=True,
            copy=False,
        )

        sample_weight = np.concatenate(weight_parts)

        del train_parts
        del weight_parts
        del data_part
        del part_weights
        gc.collect()

    if positive_only:
        positive_mask = train_data["target"].gt(0).to_numpy()

        if sample_weight is not None:
            sample_weight = sample_weight[positive_mask]

        train_data = train_data.loc[positive_mask].reset_index(drop=True)

    if validation:
        valid_data = pd.read_parquet(
            validation_file,
            columns=["user_id"] + features + ["target"],
        )
    else:
        valid_data = pd.read_parquet(
            validation_file,
            columns=["user_id"] + features,
        )

    print(f"Train: {train_data.shape}")
    print(f"Validation/Test: {valid_data.shape}")

    submission_user_id = valid_data["user_id"].copy()

    train_data[features] = train_data[features].astype("float32")
    valid_data[features] = valid_data[features].astype("float32")

    if use_user_id:
        train_data["user_id"] = train_data["user_id"].astype(str)
        valid_data["user_id"] = valid_data["user_id"].astype(str)

    y_train = train_data["target"].to_numpy(dtype="float32")

    if classifier:
        y_train_class = (y_train > 0).astype("int8")

        if validation:
            y_valid = valid_data["target"].to_numpy(dtype="float32")
            y_valid_class = (y_valid > 0).astype("int8")

            model.fit(
                train_data[model_features],
                y_train_class,
                eval_set=(valid_data[model_features], y_valid_class),
                early_stopping_rounds=150,
                use_best_model=True,
                verbose=verbose,
                sample_weight=sample_weight,
                cat_features=cat_features,
            )
        else:
            model.fit(
                train_data[model_features],
                y_train_class,
                verbose=verbose,
                sample_weight=sample_weight,
                cat_features=cat_features,
            )

        prediction = model.predict_proba(valid_data[model_features])[:, 1]

        submission = pd.DataFrame({
            "user_id": submission_user_id,
            "predict": prediction,
        })

        if validation:
            metrics = {
                "auc": roc_auc_score(y_valid_class, prediction),
                "logloss": log_loss(y_valid_class, prediction),
            }

            if return_details:
                submission["target"] = y_valid
                submission["target_positive"] = y_valid_class

            return submission, metrics

        return submission

    y_train_log = np.log1p(y_train)

    if validation:
        y_valid = valid_data["target"].to_numpy(dtype="float32")
        y_valid_log = np.log1p(y_valid)
        model.fit(
            train_data[model_features],
            y_train_log,
            eval_set=(valid_data[model_features], y_valid_log),
            early_stopping_rounds=150,
            use_best_model=True,
            verbose=verbose,
            sample_weight=sample_weight,
            cat_features=cat_features,
        )
    else:
        model.fit(
            train_data[model_features],
            y_train_log,
            verbose=verbose,
            sample_weight=sample_weight,
            cat_features=cat_features,
        )

    prediction_log = model.predict(
        valid_data[model_features]
    )

    prediction_log = np.maximum(prediction_log, 0)
    prediction = np.expm1(prediction_log)

    submission = pd.DataFrame({
        "user_id": submission_user_id,
        "predict": prediction,
    })

    if validation:
        if return_details:
            submission["target"] = y_valid
            submission["predict_log"] = prediction_log

        catboost_rmsle = root_mean_squared_log_error(
            y_valid,
            prediction,
        )

        return submission, catboost_rmsle

    return submission


def test_new_experiment(
        model,
        train_files,
        selected_features=None,
        sample_weight=None,
        use_user_id=False,
        verbose=100,
        baseline=None,
        min_improvement=0.001,
        save_fold_predictions=False,
        output_dir="result_on_fold",
        output_file="catboost_fold_predictions.csv",
        classifier=False,
        positive_only=False,
):
    result = []
    logloss_result = []

    if save_fold_predictions:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)

    for validation_index in range(2, len(train_files)):
        submission, metric = train_model_and_validation_catboost(
            clone(model),
            train_files[:validation_index],
            train_files[validation_index],
            selected_features=selected_features,
            sample_weight=sample_weight,
            use_user_id=use_user_id,
            verbose=verbose,
            return_details=save_fold_predictions,
            classifier=classifier,
            positive_only=positive_only,
        )

        if save_fold_predictions:
            fold = validation_index - 1
            validation_file = train_files[validation_index]
            snapshot_name = os.path.basename(validation_file).replace(".parquet", "")

            if classifier:
                fold_predictions = pd.DataFrame({
                    "user_id": submission["user_id"].to_numpy(),
                    "fold": fold,
                    "snapshot_name": snapshot_name,
                    "target_true": submission["target"].to_numpy(),
                    "target_positive_true": submission["target_positive"].to_numpy(),
                    "pred_positive_probability": submission["predict"].to_numpy(),
                    "auc": metric["auc"],
                    "logloss": metric["logloss"],
                })
            else:
                fold_predictions = pd.DataFrame({
                    "user_id": submission["user_id"].to_numpy(),
                    "fold": fold,
                    "snapshot_name": snapshot_name,
                    "target_true": submission["target"].to_numpy(),
                    "pred_log": submission["predict_log"].to_numpy(),
                    "pred_expm1": submission["predict"].to_numpy(),
                    "rmsle": metric,
                })

            if fold == 1:
                fold_predictions.to_csv(
                    output_path,
                    index=False,
                )
            else:
                fold_predictions.to_csv(
                    output_path,
                    mode="a",
                    header=False,
                    index=False,
                )

            print(f"Saved fold {fold} predictions: {output_path}")

        del submission
        gc.collect()
        if classifier:
            result.append(metric["auc"])
            logloss_result.append(metric["logloss"])
            print(
                f"Fold {validation_index - 1}: "
                f"AUC={metric['auc']}; Logloss={metric['logloss']}"
            )
        else:
            result.append(metric)
            print(f"Fold {validation_index - 1}: RMSLE={metric}")

    if classifier:
        mean_auc = float(np.mean(result))
        mean_logloss = float(np.mean(logloss_result))
        print(f"Mean AUC={mean_auc}")
        print(f"Mean Logloss={mean_logloss}")

        return {
            "auc": mean_auc,
            "logloss": mean_logloss,
        }

    mean = float(np.mean(result))
    print(f"Mean RMSLE={mean}")

    if baseline is not None:
        if baseline - mean >= min_improvement:
            return mean

        return False

    return mean
