"""Train QuEEN-DTA on Davis and KIBA (fixed test fold, 5 validation folds, 200 epochs).

    python scripts/run_benchmark.py --datasets davis kiba --folds 0 1 2 3 4

Finished runs are saved in results/ and skipped on a re-run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from queen_dta import config as P  # noqa: E402
from queen_dta.reporting import main as report  # noqa: E402
from queen_dta.trainer import run_dataset  # noqa: E402
from queen_dta.utils import check_gpu, keep_awake, release_gpu, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(P.DATASETS), choices=P.DATASETS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(P.N_FOLDS)))
    args = parser.parse_args()
    bad = [f for f in args.folds if not 0 <= f < P.N_FOLDS]
    if bad:
        raise SystemExit(f"folds must be 0..{P.N_FOLDS - 1}, got {bad}")

    release_gpu()
    set_seed(P.SEED_BASE)
    check_gpu()
    with keep_awake():
        for dataset in args.datasets:
            run_dataset(dataset, args.folds)
    report()


if __name__ == "__main__":
    main()
