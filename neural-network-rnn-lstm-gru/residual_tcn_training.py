import gc
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from residual_tcn_dataset import (
    create_residual_dataloader,
    read_residual_for_sequence,
    read_sequence_for_prediction,
    read_sequence_with_residual,
)


def train_tcn_with_validation(
        model,
        train_files,
        valid_file,
        residual_file,
        feature_columns,
        target_column,
        epochs,
        device,
        patience=3,
        batch_size=128,
        learning_rate=0.001,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    best_model = deepcopy(model.state_dict())
    best_valid_loss = np.inf
    epochs_without_improvement = 0

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        train_users = 0

        for train_file in train_files:
            print(f"Загружаем train {train_file}")
            train_data = read_sequence_with_residual(
                sequence_file=train_file,
                residual_file=residual_file,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            train_loader = create_residual_dataloader(
                data=train_data,
                feature_columns=feature_columns,
                target_column=target_column,
                batch_size=batch_size,
                shuffle=True,
            )

            for features, target, user_id in tqdm(
                    train_loader,
                    desc="Train batches",
                    leave=False,
            ):
                features = features.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                prediction = model(features)
                loss = criterion(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item() * len(target)
                train_users += len(target)

            del train_loader
            del train_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss = train_loss_sum / train_users
        train_losses.append(train_loss)

        valid_loss = calculate_residual_loss(
            model=model,
            sequence_file=valid_file,
            residual_file=residual_file,
            feature_columns=feature_columns,
            target_column=target_column,
            device=device,
            batch_size=batch_size,
        )
        valid_losses.append(valid_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train residual RMSE={train_loss ** 0.5:.5f} | "
            f"valid residual RMSE={valid_loss ** 0.5:.5f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Остановка на эпохе {epoch + 1}")
            break

    model.load_state_dict(best_model)
    model.eval()

    best_epoch = int(np.argmin(valid_losses) + 1)
    return best_epoch, model, train_losses, valid_losses


def calculate_residual_loss(
        model,
        sequence_file,
        residual_file,
        feature_columns,
        target_column,
        device,
        batch_size=128,
):
    criterion = nn.MSELoss()
    model.eval()

    valid_data = read_sequence_with_residual(
        sequence_file=sequence_file,
        residual_file=residual_file,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    valid_loader = create_residual_dataloader(
        data=valid_data,
        feature_columns=feature_columns,
        target_column=target_column,
        batch_size=batch_size,
        shuffle=False,
    )

    loss_sum = 0
    users_count = 0

    with torch.inference_mode():
        for features, target, user_id in tqdm(
                valid_loader,
                desc="Validation batches",
                leave=False,
        ):
            features = features.to(device)
            target = target.to(device)

            prediction = model(features)
            loss = criterion(prediction, target)

            loss_sum += loss.item() * len(target)
            users_count += len(target)

    del valid_loader
    del valid_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return loss_sum / users_count


def fit_tcn_on_all_train(
        model,
        train_files,
        residual_file,
        feature_columns,
        target_column,
        epochs,
        device,
        batch_size=128,
        learning_rate=0.001,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        train_users = 0

        for train_file in train_files:
            print(f"Загружаем train {train_file}")
            train_data = read_sequence_with_residual(
                sequence_file=train_file,
                residual_file=residual_file,
                feature_columns=feature_columns,
                target_column=target_column,
            )
            train_loader = create_residual_dataloader(
                data=train_data,
                feature_columns=feature_columns,
                target_column=target_column,
                batch_size=batch_size,
                shuffle=True,
            )

            for features, target, user_id in tqdm(
                    train_loader,
                    desc="Train batches",
                    leave=False,
            ):
                features = features.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                prediction = model(features)
                loss = criterion(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item() * len(target)
                train_users += len(target)

            del train_loader
            del train_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss = train_loss_sum / train_users
        train_losses.append(train_loss)

        print(
            f"Final epoch {epoch + 1}/{epochs} | "
            f"train residual RMSE={train_loss ** 0.5:.5f}"
        )

    model.eval()
    return model, train_losses


def predict_residual(
        model,
        sequence_file,
        feature_columns,
        device,
        batch_size=128,
):
    data = read_sequence_for_prediction(
        sequence_file=sequence_file,
        feature_columns=feature_columns,
    )
    loader = create_residual_dataloader(
        data=data,
        feature_columns=feature_columns,
        target_column=None,
        batch_size=batch_size,
        shuffle=False,
    )

    model = model.to(device)
    model.eval()

    predictions = []
    user_ids = []

    with torch.inference_mode():
        for features, target, batch_user_ids in tqdm(
                loader,
                desc="Predict batches",
                leave=False,
        ):
            features = features.to(device)
            prediction = model(features)

            predictions.append(prediction.cpu().numpy())
            user_ids.append(batch_user_ids.cpu().numpy())

    result = pd.DataFrame(
        {
            "user_id": np.concatenate(user_ids),
            "tcn_residual_prediction": np.concatenate(predictions),
        }
    )

    del loader
    del data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def predict_residual_validation(
        model,
        sequence_file,
        residual_file,
        feature_columns,
        target_column,
        device,
        batch_size=128,
):
    prediction = predict_residual(
        model=model,
        sequence_file=sequence_file,
        feature_columns=feature_columns,
        device=device,
        batch_size=batch_size,
    )

    residual = read_residual_for_sequence(
        sequence_file=sequence_file,
        residual_file=residual_file,
        target_column=target_column,
    )

    prediction = prediction.merge(
        residual[
            [
                "user_id",
                "target",
                "target_log",
                "base_hurdle_log",
                target_column,
            ]
        ],
        on="user_id",
        how="inner",
    )

    return prediction


def score_residual_alpha(validation_prediction, alpha):
    final_log = (
        validation_prediction["base_hurdle_log"]
        + alpha * validation_prediction["tcn_residual_prediction"]
    )
    final_log = np.maximum(final_log, 0)

    error = validation_prediction["target_log"] - final_log
    rmsle = np.sqrt(np.mean(error**2))

    return float(rmsle)


def find_best_alpha(validation_prediction, alphas):
    rows = []

    for alpha in alphas:
        rows.append(
            {
                "alpha": alpha,
                "rmsle": score_residual_alpha(
                    validation_prediction=validation_prediction,
                    alpha=alpha,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("rmsle").reset_index(drop=True)
