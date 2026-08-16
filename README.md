# Ozon Tech E-CUP 2026 — ценность пользователей поиска

Решение задачи прогнозирования GMV пользователя на следующие 30 дней. Метрика соревнования — RMSLE.

## Структура

- `boosting/` — создание классических срезов, CatBoost и LightGBM.
- `neural-network-rnn-lstm-gru/` — sequence-срезы, признаки, Dataset и модели RNN/GRU/LSTM.
- `ensemble/` — подбор весов ансамбля, сохранённые предсказания и готовые сабмиты.
- `FEATURES_README.md` — описание построенных признаков.

Исходные parquet-файлы и обученные веса моделей не хранятся в GitHub. Пути к исходному `train.parquet` задаются в скриптах построения датасетов.

## Лучший результат

Лучший сабмит получен ансамблем в log-пространстве:

- CatBoost: `0.67497`
- LightGBM: `0.18944`
- Simple GRU: `0.13560`
- validation RMSLE: `1.740428`
- leaderboard RMSLE: `1.655556`

Готовый файл: `ensemble/submissions/submission_catboost_lgbm_gru.csv`.

## Порядок запуска

1. `boosting/build_classic_snapshots.py` — подготовка табличных срезов.
2. `boosting/main.ipynb` — обучение CatBoost/LightGBM и сохранение предсказаний.
3. `neural-network-rnn-lstm-gru/build_sequence_snapshot.py` — sequence-срезы.
4. `neural-network-rnn-lstm-gru/feature_engineering.py` — sequence-признаки.
5. `neural-network-rnn-lstm-gru/gru_predict.ipynb` — финальное обучение Simple GRU.
6. `ensemble/ensemble_test.ipynb` — подбор весов и итоговый ансамбль.
