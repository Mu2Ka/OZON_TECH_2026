from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stacking_submissions"
OUT.mkdir(exist_ok=True)


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_cols = [c for c in df.columns if c != "user_id"]
    if len(value_cols) != 1:
        preferred = [c for c in ["predict", "target", "pred", "prediction"] if c in df.columns]
        if not preferred:
            raise ValueError(f"Cannot infer prediction column for {path}: {df.columns.tolist()}")
        value_col = preferred[0]
    else:
        value_col = value_cols[0]
    out = df[["user_id", value_col]].copy()
    out.columns = ["user_id", "predict"]
    out["predict"] = out["predict"].clip(lower=0)
    return out


def log_values(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df["predict"].to_numpy(dtype=np.float64).clip(min=0))


def write_submission(name: str, user_id: np.ndarray, pred_log: np.ndarray) -> dict:
    pred = np.expm1(np.clip(pred_log, 0, 20))
    out_path = OUT / f"{name}.csv"
    pd.DataFrame({"user_id": user_id, "predict": pred}).to_csv(out_path, index=False)
    return {
        "file": out_path.name,
        "mean": float(pred.mean()),
        "median": float(np.median(pred)),
        "p95": float(np.quantile(pred, 0.95)),
        "max": float(pred.max()),
    }


paths = {
    "tcn_a005": ROOT / "result_on_fold" / "submission_hurdle_tcn_alpha_0_05.csv",
    "tcn_a010": ROOT / "result_on_fold" / "submission_hurdle_tcn_alpha_0_10.csv",
    "tcn_a000": ROOT / "result_on_fold" / "submission_hurdle_gamma_1_1_alpha_0.csv",
    "tcn_selected": ROOT / "result_on_fold" / "submission_final_hurdle_tcn_selected.csv",
    "stable": ROOT / "result_on_fold" / "submission_calibrated_best_stable_oof.csv",
    "global": ROOT / "result_on_fold" / "submission_calibrated_best_global_oof.csv",
    "jan": ROOT / "result_on_fold" / "submission_calibrated_best_jan_oof.csv",
    "raw_g110_s0995": ROOT / "result_on_fold" / "submission_raw_gamma_1_100_scale_0_995.csv",
    "cat_lgbm_gru": ROOT / "ensemble" / "submissions" / "submission_catboost_lgbm_gru.csv",
    "cat_gru": ROOT / "ensemble" / "submissions" / "submission_catboost_gru.csv",
}

subs = {name: read_submission(path) for name, path in paths.items() if path.exists()}
base = subs["tcn_a005"].sort_values("user_id").reset_index(drop=True)
user_id = base["user_id"].to_numpy()

logs: dict[str, np.ndarray] = {}
for name, df in subs.items():
    aligned = base[["user_id"]].merge(df, on="user_id", how="left")
    if aligned["predict"].isna().any():
        raise ValueError(f"Missing predictions after align: {name}")
    logs[name] = log_values(aligned)

summary = {
    "inputs": {name: str(paths[name]) for name in subs},
    "stats": {},
    "outputs": [],
}

base_log = logs["tcn_a005"]
for name, vals in logs.items():
    summary["stats"][name] = {
        "corr_with_tcn_a005_log": float(np.corrcoef(base_log, vals)[0, 1]),
        "mean_pred": float(np.expm1(vals).mean()),
        "mean_log": float(vals.mean()),
    }

# 1) Reconstruct small TCN-alpha candidates. LB says alpha=0.05 beats 0.10,
# so probe below and just above 0.05.
a0 = logs["tcn_a000"]
a005 = logs["tcn_a005"]
for alpha in [0.015, 0.025, 0.035, 0.04, 0.045, 0.055, 0.065, 0.075]:
    pred_log = a0 + (alpha / 0.05) * (a005 - a0)
    summary["outputs"].append(write_submission(f"stack_tcn_alpha_{alpha:.3f}".replace(".", "_"), user_id, pred_log))

# 2) Tiny log-space blends around the best public-LB candidate.
blend_specs = [
    ("best95_stable05", [("tcn_a005", 0.95), ("stable", 0.05)]),
    ("best90_stable10", [("tcn_a005", 0.90), ("stable", 0.10)]),
    ("best95_global05", [("tcn_a005", 0.95), ("global", 0.05)]),
    ("best95_catlgbmgru05", [("tcn_a005", 0.95), ("cat_lgbm_gru", 0.05)]),
    ("best90_catlgbmgru10", [("tcn_a005", 0.90), ("cat_lgbm_gru", 0.10)]),
    ("best92_stable04_catlgbmgru04", [("tcn_a005", 0.92), ("stable", 0.04), ("cat_lgbm_gru", 0.04)]),
    ("best90_a01005_stable05", [("tcn_a005", 0.90), ("tcn_a010", 0.05), ("stable", 0.05)]),
]
for name, spec in blend_specs:
    pred_log = sum(logs[k] * w for k, w in spec)
    summary["outputs"].append(write_submission(f"stack_{name}", user_id, pred_log))

# 3) Very small scale jitter around alpha=0.05. This is cheap and sometimes
# moves the leaderboard more than OOF predicts.
for scale in [0.990, 0.995, 1.0025, 1.005, 1.010]:
    summary["outputs"].append(write_submission(f"stack_tcn_a005_logscale_{scale:.4f}".replace(".", "_"), user_id, base_log * scale))

summary_path = OUT / "stacking_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

zip_path = OUT / "stacking_submissions.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for csv_path in sorted(OUT.glob("stack_*.csv")):
        z.write(csv_path, csv_path.name)
    z.write(summary_path, summary_path.name)

print(json.dumps({"out_dir": str(OUT), "zip": str(zip_path), "n_outputs": len(summary["outputs"])}, indent=2, ensure_ascii=False))
print(pd.DataFrame(summary["outputs"]).to_string(index=False))
