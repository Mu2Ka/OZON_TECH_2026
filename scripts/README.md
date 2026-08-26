# Скрипты

Здесь лежат только вспомогательные скрипты, которые были нужны для финальных экспериментов и разбора результатов.

## `stacking/`

- `build_clean_stacking_dataset.py` — собирает чистый датасет для метамодели из прогнозов CatBoost, XGBoost и LightGBM.
- `build_stacking_dataset_with_base_features.py` — добавляет к датасету метамодели небольшой набор базовых признаков пользователя.

## `ensembling/`

- `calibrate_hurdle_submissions.py` — подбирает калибровки для двухэтапной CatBoost-модели.
- `make_final_tcn_submission.py` — собирает финальный ансамбль CatBoost + TCN.
- `stack_submissions.py` — простой скрипт для проверки небольших ансамблей из готовых прогнозов.

## `analysis/`

- `analyze_calendar_analog.py` — проверяет календарные гипотезы и сдвиги по времени.
- `build_cnn_residual_dataset.py` — готовит остатки ошибки для TCN-экспериментов.
