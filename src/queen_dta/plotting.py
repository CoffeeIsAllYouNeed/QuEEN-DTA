"""Comparison bar charts (MSE, CI, r_m^2) of QuEEN-DTA against the published baselines."""
import json
import re

from queen_dta import config as P

METRICS = (
    # (key in published_baselines.json position, column in scores_summary.csv, file tag, axis label)
    (1, "test_mse_mean", "mse", "MSE"),
    (0, "test_ci_mean", "ci", "CI"),
    (2, "test_r2m_mean", "rm2", r"$r_m^2$"),
)
BASELINE_COLOR = "#000080"
QUEEN_COLOR = "#006400"


def clean_name(name):
    """'DeepDTA [29]' -> 'DeepDTA'."""
    return re.sub(r"\s*\[\d+\]$", "", name)


def select_baselines(published, dataset_label, queen_mse):
    """Baselines with a higher MSE than QuEEN-DTA, minus P.PLOT_EXCLUDE, in table order."""
    chosen = []
    for name, (_, mse, _) in published[dataset_label].items():
        name = clean_name(name)
        if mse > queen_mse and name not in P.PLOT_EXCLUDE:
            chosen.append(name)
    return chosen


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    summary = pd.read_csv(P.RESULTS_DIR / "scores_summary.csv")
    published = json.loads(P.BASELINES_FILE.read_text())
    P.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for _, queen in summary.iterrows():
        label = queen["dataset"]
        tag = label.lower()
        by_name = {clean_name(k): v for k, v in published[label].items()}
        baselines = select_baselines(published, label, round(queen["test_mse_mean"], P.THRESHOLD_DECIMALS))
        names = baselines + ["QuEEN-DTA"]
        for position, column, short, axis_label in METRICS:
            values = [by_name[n][position] for n in baselines] + [queen[column]]
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            colors = [QUEEN_COLOR if n == "QuEEN-DTA" else BASELINE_COLOR for n in names]
            bars = ax.bar(range(len(names)), values, color=colors, width=0.75)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + max(values) * 0.012,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=14, fontweight="bold",
                )
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=14)
            ax.tick_params(axis="y", labelsize=14)
            ax.set_ylabel(axis_label, fontsize=20)
            ax.set_ylim(0, max(values) * 1.12)
            fig.tight_layout()
            path = P.FIGURES_DIR / f"{tag}_{short}_comparison.png"
            fig.savefig(path)
            plt.close(fig)
            print("saved", path.relative_to(P.ROOT))


if __name__ == "__main__":
    main()
