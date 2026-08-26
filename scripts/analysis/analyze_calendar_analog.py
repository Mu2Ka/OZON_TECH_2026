from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


RAW_DATA = Path(r"C:\Users\myska\Downloads\train.parquet")
CLASSIFIER_OOF = Path(
    r"C:\Users\myska\OneDrive\Рабочий стол\catboost_classifier_fold_predictions.csv"
)
POSITIVE_OOF = Path(r"C:\Users\myska\Downloads\catboost_user_errors_1.csv")
CLASSIFIER_TEST = Path(
    r"C:\Users\myska\OneDrive\Рабочий стол\catboost_classifier_test_probability.csv"
)
POSITIVE_TEST = Path(
    r"C:\Users\myska\Downloads\Telegram Desktop\catboost_competition_test_predictions_1.csv"
)

OUTPUT_DIR = Path("result_on_fold/calendar_analog")
JAN_SNAPSHOT = "train_jan_predict_from_2026-01-14"


def rmsle_log(target_log, prediction_log):
    return float(np.sqrt(np.mean(np.square(target_log - prediction_log))))


def aggregate_calendar_windows():
    raw_path = str(RAW_DATA).replace("\\", "/").replace("'", "''")
    query = f"""
        SELECT
            user_id,
            SUM(CASE WHEN event_date BETWEEN DATE '2025-01-15' AND DATE '2025-02-13'
                     THEN gmv ELSE 0 END) AS jan_target_analog,
            SUM(CASE WHEN event_date BETWEEN DATE '2025-01-01' AND DATE '2025-01-14'
                     THEN gmv ELSE 0 END) AS jan_growth_reference_2025,
            SUM(CASE WHEN event_date BETWEEN DATE '2026-01-01' AND DATE '2026-01-14'
                     THEN gmv ELSE 0 END) AS jan_growth_reference_2026,
            SUM(CASE WHEN event_date BETWEEN DATE '2025-02-14' AND DATE '2025-03-15'
                     THEN gmv ELSE 0 END) AS competition_target_analog,
            SUM(CASE WHEN event_date BETWEEN DATE '2025-01-31' AND DATE '2025-02-13'
                     THEN gmv ELSE 0 END) AS competition_growth_reference_2025,
            SUM(CASE WHEN event_date BETWEEN DATE '2026-01-31' AND DATE '2026-02-13'
                     THEN gmv ELSE 0 END) AS competition_growth_reference_2026
        FROM read_parquet('{raw_path}')
        GROUP BY user_id
        ORDER BY user_id
    """
    return duckdb.sql(query).df()


def load_january_oof():
    classifier = pd.read_csv(
        CLASSIFIER_OOF,
        usecols=[
            "user_id",
            "snapshot_name",
            "target_true",
            "pred_positive_probability",
        ],
    )
    classifier = classifier.loc[classifier["snapshot_name"].eq(JAN_SNAPSHOT)]
    classifier = classifier.rename(
        columns={
            "target_true": "target",
            "pred_positive_probability": "probability",
        }
    )

    positive = pd.read_csv(
        POSITIVE_OOF,
        usecols=["user_id", "snapshot_name", "target_true", "pred_expm1"],
    )
    positive = positive.loc[positive["snapshot_name"].eq(JAN_SNAPSHOT)]
    positive = positive.rename(
        columns={"target_true": "positive_target", "pred_expm1": "positive_prediction"}
    )

    data = classifier.merge(
        positive[["user_id", "positive_target", "positive_prediction"]],
        on="user_id",
        how="inner",
        validate="one_to_one",
    )
    if not np.allclose(data["target"], data["positive_target"]):
        raise ValueError("January targets differ between classifier and regressor OOF")
    return data.drop(columns="positive_target")


def load_competition_predictions():
    classifier = pd.read_csv(CLASSIFIER_TEST, usecols=["user_id", "predict"])
    classifier = classifier.rename(columns={"predict": "probability"})
    positive = pd.read_csv(POSITIVE_TEST, usecols=["user_id", "pred_expm1"])
    positive = positive.rename(columns={"pred_expm1": "positive_prediction"})
    return classifier.merge(positive, on="user_id", how="inner", validate="one_to_one")


def make_analog_log(data, target_column, old_reference, new_reference, alpha, clip, k):
    target_analog_log = np.log1p(data[target_column].to_numpy(dtype=np.float64))
    old_value = data[old_reference].to_numpy(dtype=np.float64)
    new_value = data[new_reference].to_numpy(dtype=np.float64)
    growth = np.log1p(new_value) - np.log1p(old_value)
    growth = np.clip(growth, -clip, clip)
    support = (old_value + new_value) / (old_value + new_value + k)
    return np.maximum(target_analog_log + alpha * support * growth, 0.0)


def base_hurdle_log(data, gamma):
    probability = data["probability"].to_numpy(dtype=np.float64).clip(0, 1)
    positive_log = np.log1p(
        data["positive_prediction"].to_numpy(dtype=np.float64).clip(min=0)
    )
    return np.power(probability, gamma) * positive_log


def tune_on_january(data):
    target_log = np.log1p(data["target"].to_numpy(dtype=np.float64))
    rows = []

    for gamma in np.round(np.arange(0.90, 1.151, 0.01), 2):
        base_log = base_hurdle_log(data, gamma)
        base_score = rmsle_log(target_log, base_log)

        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            for clip in (0.5, 1.0, 1.5, 2.0):
                for k in (10.0, 50.0, 100.0, 250.0):
                    analog_log = make_analog_log(
                        data,
                        "jan_target_analog",
                        "jan_growth_reference_2025",
                        "jan_growth_reference_2026",
                        alpha,
                        clip,
                        k,
                    )
                    analog_score = rmsle_log(target_log, analog_log)

                    for weight in np.round(np.arange(0.05, 0.301, 0.025), 3):
                        prediction_log = (1 - weight) * base_log + weight * analog_log
                        rows.append(
                            {
                                "gamma": gamma,
                                "alpha": alpha,
                                "clip": clip,
                                "k": k,
                                "weight": weight,
                                "base_rmsle": base_score,
                                "analog_rmsle": analog_score,
                                "blend_rmsle": rmsle_log(target_log, prediction_log),
                            }
                        )

    return pd.DataFrame(rows).sort_values("blend_rmsle").reset_index(drop=True)


def create_submissions(test, best_row):
    gamma = float(best_row["gamma"])
    alpha = float(best_row["alpha"])
    clip = float(best_row["clip"])
    k = float(best_row["k"])

    base_log = base_hurdle_log(test, gamma)
    analog_log = make_analog_log(
        test,
        "competition_target_analog",
        "competition_growth_reference_2025",
        "competition_growth_reference_2026",
        alpha,
        clip,
        k,
    )

    created = []
    for weight in (0.05, 0.10, 0.15, 0.20, 0.25):
        prediction_log = (1 - weight) * base_log + weight * analog_log
        submission = pd.DataFrame(
            {
                "user_id": test["user_id"].to_numpy(),
                "predict": np.expm1(np.maximum(prediction_log, 0)),
            }
        )
        output = OUTPUT_DIR / f"submission_calendar_analog_w{weight:.2f}.csv"
        submission.to_csv(output, index=False)
        created.append(output)
    return created


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calendar = aggregate_calendar_windows()
    calendar.to_parquet(OUTPUT_DIR / "calendar_windows.parquet", index=False)

    january = load_january_oof().merge(
        calendar,
        on="user_id",
        how="left",
        validate="one_to_one",
    )
    january = january.fillna(0)
    tuning = tune_on_january(january)
    tuning.to_csv(OUTPUT_DIR / "calendar_analog_january_tuning.csv", index=False)

    best = tuning.iloc[0]
    competition = load_competition_predictions().merge(
        calendar,
        on="user_id",
        how="left",
        validate="one_to_one",
    )
    competition = competition.fillna(0)
    submissions = create_submissions(competition, best)

    print("Best January parameters:")
    print(best.to_string())
    print("Top robust neighborhood:")
    print(tuning.head(20).to_string(index=False))
    print("Created submissions:")
    for path in submissions:
        print(path)


if __name__ == "__main__":
    main()
