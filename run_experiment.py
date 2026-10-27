"""Run QuEEN-DTA on the DeepH-DTA protocol.

    python run_experiment.py --datasets davis kiba --folds 0 1 2 3 4
"""
import argparse

from queen_dta import protocol as P
from queen_dta.common import check_gpu, keep_awake, release_gpu, set_seed
from queen_dta.report import main as report
from queen_dta.train import run_dataset


def main():
    parser = argparse.ArgumentParser()
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
