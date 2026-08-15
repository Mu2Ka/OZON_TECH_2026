import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from dataloader_dataset import create_dataloader_dataset
import gc

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
            for features, target in train_loader:
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
            for features, target in valid_loader:
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