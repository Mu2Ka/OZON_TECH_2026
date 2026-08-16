import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from dataloader_dataset import create_dataloader_dataset
import gc
from tqdm.auto import tqdm

epochs = 100
patience = 10


def pytorch_model_validation(
        model,
        train_files,
        valid_files,
        epochs,
        device,
        patience
):
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_losses = []
    valid_losses = []
    model = model.to(device)
    best_valid_loss = np.inf
    best_model = deepcopy(model.state_dict())
    epoch_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        train_users = 0
        for train_file in train_files:
            print(f"Загружаем {train_file}")
            train_data = pd.read_parquet(train_file)
            train_loader = create_dataloader_dataset(train_data, batch_size=128, shuffle=True)
            for features, target in tqdm(
                train_loader,
                desc="Train batches",
                leave=False,
            ):
                features = features.to(device)
                target = target.to(device)
                target_log = torch.log1p(target).to(device)
                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, target_log)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                epoch_train_loss += loss.item() * len(target)
                train_users += len(target)
            del train_loader
            del train_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_train_loss = epoch_train_loss / train_users
        train_losses.append(epoch_train_loss)
        model.eval()
        epoch_valid_loss = 0
        valid_users = 0
        valid_data = pd.read_parquet(valid_files)
        valid_loader = create_dataloader_dataset(valid_data, batch_size=128, shuffle=False)

        with torch.no_grad():
            for features, target in tqdm(
                valid_loader,
                desc="Validation batches",
                leave=False,
            ):
                features = features.to(device)
                target = target.to(device)
                target_log = torch.log1p(target).to(device)
                output = model(features)
                loss = criterion(output, target_log)
                epoch_valid_loss += loss.item() * len(target)
                valid_users += len(target)

        epoch_valid_loss = epoch_valid_loss / valid_users
        valid_losses.append(epoch_valid_loss)

        del valid_loader
        del valid_data
        gc.collect()
        train_rmsle = epoch_train_loss ** 0.5
        valid_rmsle = epoch_valid_loss ** 0.5
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train RMSLE: {train_rmsle:.5f} | "
            f"valid RMSLE: {valid_rmsle:.5f}"
        )
        if valid_rmsle < best_valid_loss:
            best_valid_loss = valid_rmsle
            best_model = deepcopy(model.state_dict())
            epoch_without_improvement = 0
        else:
            epoch_without_improvement += 1
        if epoch_without_improvement >= patience:
            print(f"остановка на эпохе {epoch + 1}")
            break
    model.load_state_dict(best_model)
    model.eval()
    return model, train_losses, valid_losses


def pytorch_model_fit(model, train_files, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader =  create_dataloader_dataset(train_files)
    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0

        for X_batch, y_batch in tqdm(
            train_loader,
            desc="Train batches",
            leave=False,
        ):
            optimizer.zero_grad()
            predictions = model(X_batch).reshape(-1)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(train_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"Final epoch {epoch + 1:03d} | "
                f"train={train_loss:.6f}"
            )

    model.eval()
    return model, train_losses


def predict_model(model, test_files):
    test_loader = create_dataloader_dataset(test_files)
    criterion = nn.MSELoss()

    model.eval()
    test_loss_sum = 0
    predictions_list = []
    targets_list = []

    with torch.inference_mode():
        for X_batch, y_batch in tqdm(
            test_loader,
            desc="Test batches",
            leave=False,
        ):
            predictions = model(X_batch).reshape(-1)
            loss = criterion(predictions, y_batch)
            test_loss_sum += loss.item() * len(X_batch)

            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(y_batch.cpu().numpy())

    test_loss = test_loss_sum / len(test_loader.dataset)
    test_predictions = np.concatenate(predictions_list)
    test_targets = np.concatenate(targets_list)

    return test_loss, test_predictions, test_targets


def evaluate_model(model, data_file, device, batch_size=128):
    data = pd.read_parquet(data_file)
    loader = create_dataloader_dataset(
        data,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    squared_error_sum = 0
    users_count = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for features, target in tqdm(
            loader,
            desc="Evaluation batches",
            leave=False,
        ):
            features = features.to(device)
            target = target.to(device)

            target_log = torch.log1p(target)
            prediction_log = model(features)
            prediction_log = torch.clamp(prediction_log, min=0)

            squared_error_sum += torch.sum(
                (prediction_log - target_log) ** 2
            ).item()
            users_count += len(target)

            predictions.append(
                torch.expm1(prediction_log).cpu().numpy()
            )
            targets.append(target.cpu().numpy())

    rmsle = np.sqrt(squared_error_sum / users_count)

    del loader
    del data
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        rmsle,
        np.concatenate(predictions),
        np.concatenate(targets),
    )
