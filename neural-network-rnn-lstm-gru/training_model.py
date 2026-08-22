import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
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


def pytorch_model_fit_all_dataset(
        model,
        train_files,
        epochs,
        device,
        batch_size=128,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        train_users = 0

        for train_file in train_files:
            print(f"Загружаем {train_file}")
            train_data = pd.read_parquet(train_file)
            train_loader = create_dataloader_dataset(
                train_data,
                batch_size=batch_size,
                shuffle=True,
            )

            for X_batch, y_batch in tqdm(
                    train_loader,
                    desc="Train batches",
                    leave=False,
            ):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()

                predictions = model(X_batch).reshape(-1)
                loss = criterion(predictions, torch.log1p(y_batch))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item() * len(X_batch)
                train_users += len(X_batch)

            del train_loader
            del train_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss = train_loss_sum / train_users
        train_losses.append(train_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"Final epoch {epoch + 1:03d} | "
                f"train RMSLE={train_loss ** 0.5:.5f}"
            )

    model.eval()
    return model, train_losses


def predict_model(
        model,
        test_files,
        device,
        batch_size=128,
):
    test_data = pd.read_parquet(test_files)
    test_loader = create_dataloader_dataset(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )
    criterion = nn.MSELoss()
    model = model.to(device)
    model.eval()
    test_loss_sum = 0
    test_users = 0
    predictions_list = []
    targets_list = []

    with torch.inference_mode():
        for X_batch, y_batch in tqdm(
                test_loader,
                desc="Test batches",
                leave=False,
        ):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch).reshape(-1)
            predictions = torch.clamp(predictions, min=0)
            target_log = torch.log1p(y_batch)
            loss = criterion(predictions, target_log)
            test_loss_sum += loss.item() * len(X_batch)
            test_users += len(X_batch)

            predictions_list.append(torch.expm1(predictions).cpu().numpy())
            targets_list.append(y_batch.cpu().numpy())

    test_loss = test_loss_sum / test_users
    test_rmsle = test_loss ** 0.5
    test_predictions = np.concatenate(predictions_list)
    test_targets = np.concatenate(targets_list)

    del test_loader
    del test_data
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_rmsle, test_predictions, test_targets
