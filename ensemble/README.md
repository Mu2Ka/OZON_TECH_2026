# Результаты ансамбля

## Папки

- `validation_predictions/` — предсказания моделей на cutoff `2025-12-15`, использованные для подбора весов.
- `competition_predictions/` — отдельные предсказания GRU для competition-выборки.
- `submissions/` — готовые файлы для отправки.

## Результаты leaderboard

| Модели | RMSLE |
|---|---:|
| CatBoost + LightGBM | 1.656651 |
| CatBoost + GRU | 1.655916 |
| CatBoost + LightGBM + GRU | **1.655556** |

Финальные веса трёх моделей подбираются в `ensemble_test.ipynb` на валидации и применяются в log-пространстве.
