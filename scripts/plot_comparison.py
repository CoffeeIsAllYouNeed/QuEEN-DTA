"""Draw the six comparison charts (Davis/KIBA x MSE/CI/r_m^2) into assets/figures/.

    python scripts/plot_comparison.py

Needs results/scores_summary.csv (written by scripts/summarize_results.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from queen_dta.plotting import main  # noqa: E402

if __name__ == "__main__":
    main()
