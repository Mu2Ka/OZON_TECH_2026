import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
DATA_DIR = Path("../data_classic")
TEST_FILE = DATA_DIR / "test_data.parquet"  # Путь к новому датасету для теста
OOF_OUTPUT_PATH = Path("../oof_predictions_catboost.parquet")
TEST_PREDS_OUTPUT_PATH = Path("../test_predictions_catboost.parquet")

EXCLUDE_COLUMNS = {"user_id", "cutoff_date", "target"}

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
# 2. ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ И ПРЕДСКАЗАНИЯ
# ==========================================
def train_catboost_rolling_oof_and_predict(test_path=TEST_FILE):
    oof_records = []
    fold_scores = []
    test_fold_preds_log = []

    # 1. Загрузка тестового датасета
    print(f"Загружаем тестовый датасет: {test_path}")
    test_df = pd.read_parquet(test_path)

    for fold_info in SNAPSHOT_FOLDS:
        fold = fold_info["fold"]
        print(f"\n=================== START FOLD {fold} ===================")

        # Загрузка Train данных фолда
        train_dfs = [pd.read_parquet(p) for p in fold_info["train"]]
        train_data = pd.concat(train_dfs, ignore_index=True)

        feature_cols = [
            c for c in train_data.columns if c not in EXCLUDE_COLUMNS
        ]

        X_train = train_data[feature_cols]
        y_train_log = np.log1p(train_data["target"].values)

        # Загрузка Val данных фолда
        val_data = pd.read_parquet(fold_info["val"])
        X_val = val_data[feature_cols]
        y_val_true = val_data["target"].values
        y_val_log = np.log1p(y_val_true)

        # Данные для теста
        X_test = test_df[feature_cols]

        # Обучение CatBoost
        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=200,
        )

        model.fit(
            X_train,
            y_train_log,
            eval_set=(X_val, y_val_log),
            early_stopping_rounds=150,
            use_best_model=True,
        )

        # 2. Валидационные предсказания (OOF)
        val_pred_log = model.predict(X_val)
        val_pred_log = np.clip(val_pred_log, a_min=0, a_max=None)
        val_pred_orig = np.expm1(val_pred_log)

        fold_rmsle = root_mean_squared_error(y_val_log, val_pred_log)
        fold_scores.append(fold_rmsle)

        # 3. Тестовые предсказания для текущего фолда
        fold_test_pred_log = model.predict(X_test)
        fold_test_pred_log = np.clip(fold_test_pred_log, a_min=0, a_max=None)
        test_fold_preds_log.append(fold_test_pred_log)

        # Вывод результатов текущего фолда
        print(f"\n--- [FOLD {fold} SUMMARY] ---")
        print(f"Validation RMSLE: {fold_rmsle:.5f}")
        print(
            f"Test Preds (log) -> Mean: {fold_test_pred_log.mean():.4f} | Std: {fold_test_pred_log.std():.4f}"
        )
        print(
            f"Test Preds (orig) -> Mean: {np.expm1(fold_test_pred_log).mean():.4f}"
        )

        # Сохранение OOF фолда
        fold_oof_df = pd.DataFrame(
            {
                "user_id": val_data["user_id"].values,
                "cutoff": val_data["cutoff_date"].values,
                "target": y_val_true,
                "pred_log": val_pred_log,
                "pred": val_pred_orig,
                "model": "CatBoost",
                "fold": fold,
            }
        )
        oof_records.append(fold_oof_df)

        del train_dfs, train_data, val_data, X_train, X_val, model
        gc.collect()

    # ==========================================
    # 3. АГРЕГАЦИЯ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ==========================================
    # Сохранение OOF
    total_oof_df = pd.concat(oof_records, ignore_index=True)
    total_oof_df.to_parquet(OOF_OUTPUT_PATH, index=False)

    # Усреднение предсказаний со всех фолдов для теста
    avg_test_pred_log = np.mean(test_fold_preds_log, axis=0)
    avg_test_pred_orig = np.expm1(avg_test_pred_log)

    test_result_df = pd.DataFrame(
        {
            "user_id": test_df["user_id"].values,
            "pred_log": avg_test_pred_log,
            "pred": avg_test_pred_orig,
        }
    )

    # Добавляем колонки с предсказаниями отдельных фолдов, чтобы их можно было оценить
    for i, fold_pred in enumerate(test_fold_preds_log):
        test_result_df[f"pred_log_fold_{i}"] = fold_pred

    test_result_df.to_parquet(TEST_PREDS_OUTPUT_PATH, index=False)

    print("\n=================== ИТОГИ ===================")
    print(f"Mean Fold RMSLE: {np.mean(fold_scores):.5f}")
    print(f"OOF сохранен в: {OOF_OUTPUT_PATH.resolve()}")
    print(f"Тестовые предсказания сохранены в: {TEST_PREDS_OUTPUT_PATH.resolve()}")

    return test_result_df


if __name__ == "__main__":
    # Запуск функции на новом тестовом датасете
    train_catboost_rolling_oof_and_predict(test_path=TEST_FILE)
