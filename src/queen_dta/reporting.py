"""Summary over the folds and comparison with the published baselines."""
import json

import pandas as pd

from queen_dta import config as P

METRICS = ["mse", "ci", "r2m", "r2"]


def summarise():
    paths = sorted(P.RESULTS_DIR.glob("*_fold*.json"))
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    runs = pd.DataFrame([json.loads(p.read_text()) for p in paths])
    runs["dataset"] = runs["dataset"].map(P.DATASET_LABELS)
    runs = runs.sort_values(["dataset", "val_fold"]).reset_index(drop=True)
    runs.round(4).to_csv(P.RESULTS_DIR / "scores_per_fold.csv", index=False)
    rows = []
    for dataset, group in runs.groupby("dataset", sort=False):
        row = {"dataset": dataset, "runs": len(group)}
        for part in ("val", "test"):
            for metric in METRICS:
                values = group[f"{part}_{metric}"].values
                row[f"{part}_{metric}_mean"] = values.mean()
                row[f"{part}_{metric}_std"] = values.std()
        rows.append(row)
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(P.RESULTS_DIR / "scores_summary.csv", index=False)
    return runs, summary


def compare(summary):
    published = json.loads(P.BASELINES_FILE.read_text())
    rows = []
    for _, queen in summary.iterrows():
        dataset = queen["dataset"]
        d = P.THRESHOLD_DECIMALS
        ci = round(queen["test_ci_mean"], d)
        mse = round(queen["test_mse_mean"], d)
        r2m = round(queen["test_r2m_mean"], d)
        for model, (p_ci, p_mse, p_r2m) in published[dataset].items():
            rows.append({
                "dataset": dataset,
                "model": model,
                "ci": p_ci,
                "mse": p_mse,
                "r2m": p_r2m,
                "queen_ci": ci,
                "queen_mse": mse,
                "queen_r2m": r2m,
                "queen_better_on_all": bool(ci > p_ci and mse < p_mse and r2m > p_r2m),
            })
    return pd.DataFrame(rows)


def main():
    runs, summary = summarise()
    if summary.empty:
        print("No finished runs yet")
        return
    pd.set_option("display.width", 200)
    print("Per fold (validation fold = run):")
    cols = ["dataset", "val_fold", "seed", "best_epoch", "val_mse", "test_mse", "test_ci", "test_r2m", "test_r2"]
    print(runs[cols].round(4).to_string(index=False))
    print("\nQuEEN-DTA, mean (std) over the folds, fixed test set:")
    for _, r in summary.iterrows():
        print(
            f"{r['dataset']:6s} runs={int(r['runs'])} | "
            f"CI {r['test_ci_mean']:.4f} ({r['test_ci_std']:.4f}) | "
            f"MSE {r['test_mse_mean']:.4f} ({r['test_mse_std']:.4f}) | "
            f"R2M {r['test_r2m_mean']:.4f} ({r['test_r2m_std']:.4f}) | "
            f"R2 {r['test_r2_mean']:.4f} ({r['test_r2_std']:.4f})"
        )
    if (summary["runs"] < P.N_FOLDS).any():
        print("\nWARNING: fewer than 5 folds finished for some dataset")
    comparison = compare(summary)
    comparison.to_csv(P.RESULTS_DIR / "comparison_with_published_baselines.csv", index=False)
    print("\nAgainst the published baselines (QuEEN-DTA rounded to 3 decimals, strict):")
    print(comparison.to_string(index=False))
    wins = comparison.pivot(index="model", columns="dataset", values="queen_better_on_all")
    if {"Davis", "KIBA"} <= set(wins.columns):
        keep = wins[wins["Davis"] & wins["KIBA"]].index.tolist()
        print("\nQuEEN-DTA better on CI, MSE and R2M on BOTH datasets than:")
        print(", ".join(keep) if keep else "none")


if __name__ == "__main__":
    main()
