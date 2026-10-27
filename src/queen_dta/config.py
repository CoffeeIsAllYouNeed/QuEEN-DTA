"""Benchmark protocol and QuEEN-DTA hyper-parameters. Every setting lives here.

Benchmark protocol: the DeepDTA data and folds, 5 runs, 200 epochs.

What the protocol fixes
  - Data and split: DeepDTA files. One fixed test fold (Davis 5,010 pairs, KIBA 19,709
    pairs) and 5 training folds. Davis 20,037 train / 5,009 validation /
    5,010 test, KIBA 78,836 / 19,709 / 19,709. That is 4 of the 5 folds for training and
    1 fold for validation. The paper runs this over the 5 folds ("5-fold cross validation")
    and mean and std are reported on the fixed test fold.
  - 200 training epochs.

What the protocol does NOT state (so it is set here and documented)
  - Random seed: none. Run k uses SEED_BASE + k.
  - Which weights are tested: not stated. The default is the
    weights of the epoch with the best validation MSE. Set SELECT_BEST_VALIDATION = False
    to test the weights of the last epoch instead.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "assets" / "figures"
BASELINES_FILE = ROOT / "benchmarks" / "published_baselines.json"

DATASETS = ("davis", "kiba")
DATASET_LABELS = {"davis": "Davis", "kiba": "KIBA"}
TRAIN_FOLD_FILE = "folds/train_fold_setting1.txt"
TEST_FOLD_FILE = "folds/test_fold_setting1.txt"
N_FOLDS = 5
EXPECTED_COUNTS = {"davis": (25046, 5010), "kiba": (98545, 19709)}
THRESHOLD_DECIMALS = 3

MAX_EPOCHS = 200
SEED_BASE = 0
SELECT_BEST_VALIDATION = True
PATIENCE = None

# Reference optimiser settings: batch 256, Adam, lr 1e-5, dropout 0.1, MSE loss.
# False keeps the tuned QuEEN-DTA settings of the QuEEN-DTA paper.
MATCH_REFERENCE_OPTIMISER = False
QUEEN_BATCH_SIZE = 256 if MATCH_REFERENCE_OPTIMISER else 64
QUEEN_LEARNING_RATE = 1e-5 if MATCH_REFERENCE_OPTIMISER else 0.00013068942544389324
QUEEN_DROPOUT = 0.1 if MATCH_REFERENCE_OPTIMISER else 0.29038004945599566
QUEEN_WEIGHT_DECAY = 0.0 if MATCH_REFERENCE_OPTIMISER else 0.004
QUEEN_LOSS = "mse" if MATCH_REFERENCE_OPTIMISER else "huber"

# Figures: published baselines left out of the comparison charts (besides those whose MSE
# is not higher than QuEEN-DTA's, which are always left out).
PLOT_EXCLUDE = ("GANsDTA", "DeepGS")
