from copy import deepcopy
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

# Импортируем твою функцию создания DataLoader
from dataloader_dataset import create_dataloader_dataset

# ==========================================
# 1. КОНФИГУРАЦИЯ И НАСТРОЙКИ
# ==========================================
DATA_DIR = Path("../data_classic")
OOF_OUTPUT_PATH = Path("../oof_predictions_pytorch.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 128

# Файлы срезов для Rolling Validation
SNAPSHOT_FOLDS = [
    {
        "fold": 0,
        "train": [
            DATA_DIR / "train_aug_predict_from_2025-08-17.parquet",
            DATA_DIR / "train_sep_predict_from_2025-09-16.parquet",
        ],
        "val": DATA_DIR / "train_1_predict_from_2025-10-16.parquet",
    },
    {
        "fold": 1,
        "train": [
            DATA_DIR / "train_sep_predict_from_2025-09-16.parquet",
            DATA_DIR / "train_1_predict_from_2025-10-16.parquet",
        ],
        "val": DATA_DIR / "train_2_predict_from_2025-11-15.parquet",
    },
    {
        "fold": 2,
        "train": [
            DATA_DIR / "train_1_predict_from_2025-10-16.parquet",
            DATA_DIR / "train_2_predict_from_2025-11-15.parquet",
        ],
        "val": DATA_DIR / "train_3_predict_from_2025-12-15.parquet",
    },
]


# ==========================================
# 2. ИСПРАВЛЕННЫЕ ФУНКЦИИ PYTORCH
# ==========================================
def train_one_epoch(
    model, train_files, optimizer, criterion, device, batch_size=128
):
    model.train()
    epoch_train_loss = 0
    train_users = 0

    for train_file in train_files:
        train_data = pd.read_parquet(train_file)
        train_loader = create_dataloader_dataset(
            train_data, batch_size=batch_size, shuffle=True
        )

        for features, target in tqdm(
            train_loader, desc="Train batches", leave=False
        ):
            features = features.to(device)
            target = target.to(device)
            target_log = torch.log1p(target)

            optimizer.zero_grad()
            output = model(features).squeeze(-1)  # Защита от broadcasting
            loss = criterion(output, target_log)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * len(target)
            train_users += len(target)

        del train_loader, train_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return epoch_train_loss / train_users


def predict_eval(model, valid_file, device, batch_size=128):
    model.eval()
    valid_data = pd.read_parquet(valid_file)
    valid_loader = create_dataloader_dataset(
        valid_data, batch_size=batch_size, shuffle=False
    )

    preds_log_list = []

    with torch.no_grad():
        for features, _ in tqdm(
            valid_loader, desc="Validation batches", leave=False
        ):
            features = features.to(device)
            output = model(features).squeeze(-1)
            output_clamped = torch.clamp(output, min=0)  # log1p target >= 0
            preds_log_list.append(output_clamped.cpu().numpy())

    del valid_loader, valid_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(preds_log_list)


# ==========================================
# 3. ОСНОВНОЙ ЦИКЛ ROLLING VALIDATION (OOF)
# ==========================================
def run_pytorch_rolling_oof(
    model_class, model_kwargs, epochs=EPOCHS, patience=PATIENCE
):
    oof_records = []
    fold_scores = []

    for fold_info in SNAPSHOT_FOLDS:
        fold = fold_info["fold"]
        print(
            f"\n=================== START PYTORCH FOLD {fold} ==================="
        )

        # 1. Инициализация модели и оптимизатора для текущего фолда
        model = model_class(**model_kwargs).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        best_valid_loss = np.inf
        best_model_weights = deepcopy(model.state_dict())
        epoch_without_improvement = 0

        # 2. Обучение с Early Stopping
        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model,
                fold_info["train"],
                optimizer,
                criterion,
                DEVICE,
                BATCH_SIZE,
            )

            # Проверка на валидации
            val_preds_log = predict_eval(
                model, fold_info["val"], DEVICE, BATCH_SIZE
            )
            val_data = pd.read_parquet(fold_info["val"])
            y_val_log = np.log1p(val_data["target"].values)

            val_rmsle = root_mean_squared_error(y_val_log, val_preds_log)
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | Train RMSLE: {train_loss**0.5:.5f} | Val RMSLE: {val_rmsle:.5f}"
            )

            if val_rmsle < best_valid_loss:
                best_valid_loss = val_rmsle
                best_model_weights = deepcopy(model.state_dict())
                epoch_without_improvement = 0
            else:
                epoch_without_improvement += 1

            if epoch_without_improvement >= patience:
                print(f"Early stopping сработал на эпохе {epoch + 1}")
                break

        # 3. Получение лучших предсказаний для OOF
        model.load_state_dict(best_model_weights)
        final_pred_log = predict_eval(
            model, fold_info["val"], DEVICE, BATCH_SIZE
        )
        final_pred_orig = np.expm1(final_pred_log)

        val_df = pd.read_parquet(fold_info["val"])

        # 4. Формирование обязательного формата OOF
        fold_oof_df = pd.DataFrame(
            {
                "user_id": val_df["user_id"].values,
                "cutoff": val_df["cutoff_date"].values,
                "target": val_df["target"].values,
                "pred_log": final_pred_log,
                "pred": final_pred_orig,
                "model": "PyTorch",
                "fold": fold,
            }
        )
        oof_records.append(fold_oof_df)
        fold_scores.append(best_valid_loss)

        del model, val_df
        gc.collect()

    # 5. Объединение и сохранение в Parquet
    total_oof_df = pd.concat(oof_records, ignore_index=True)
    total_oof_df.to_parquet(OOF_OUTPUT_PATH, index=False)

    overall_rmsle = root_mean_squared_error(
        np.log1p(total_oof_df["target"].values), total_oof_df["pred_log"].values
    )

    print("\n=================== ИТОГИ OOF (PyTorch) ===================")
    print(f"Mean Fold RMSLE: {np.mean(fold_scores):.5f}")
    print(f"Overall OOF RMSLE: {overall_rmsle:.5f}")
    print(f"Файл успешно сохранён: {OOF_OUTPUT_PATH.resolve()}")
    print(f"Всего строк в OOF: {len(total_oof_df)}")


# ==========================================
# 4. ЗАПУСК (Передай сюда свою PyTorch модель)
# ==========================================
# Пример вызова:
# run_pytorch_rolling_oof(MyModelClass, model_kwargs={"input_dim": 64})
