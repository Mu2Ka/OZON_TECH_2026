import gc

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
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
):
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
):
    result = []

    for validation_index in range(2, len(train_files)):
        submission, rmsle = train_model_and_validation_catboost(
            clone(model),
            train_files[:validation_index],
            train_files[validation_index],
            selected_features=selected_features,
            sample_weight=sample_weight,
            use_user_id=use_user_id,
            verbose=verbose,
        )
        del submission
        gc.collect()
        result.append(rmsle)
        print(f"Fold {validation_index - 1}: RMSLE={rmsle}")

    mean = float(np.mean(result))
    print(f"Mean RMSLE={mean}")

    if baseline is not None:
        if baseline - mean >= min_improvement:
            return mean

        return False

    return mean
