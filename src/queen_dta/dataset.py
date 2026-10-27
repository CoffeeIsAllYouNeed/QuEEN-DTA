"""Davis / KIBA loading and the fixed-test, 5-fold split of the DeepDTA release.

Row k of the returned frame is pair position k of np.where(~np.isnan(Y)), so the fold files
index the frame directly.
"""
import json
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from rdkit import Chem

from queen_dta.config import (
    DATA_DIR, EXPECTED_COUNTS, N_FOLDS, TEST_FOLD_FILE, TRAIN_FOLD_FILE,
)


def load_dataset(dataset):
    folder = DATA_DIR / dataset
    train_folds = json.load(open(folder / TRAIN_FOLD_FILE))
    test_fold = json.load(open(folder / TEST_FOLD_FILE))
    ligands = json.load(open(folder / "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(folder / "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(folder / "Y", "rb"), encoding="latin1")

    drugs = [
        Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        for d in ligands.keys()
    ]
    prots = [proteins[t] for t in proteins.keys()]
    if dataset == "davis":
        affinity = np.asarray([-np.log10(y / 1e9) for y in affinity])
    affinity = np.asarray(affinity)

    rows, cols = np.where(np.isnan(affinity) == False)  # noqa: E712
    df = pd.DataFrame({
        "drug_id": rows.astype(str),
        "protein_id": cols.astype(str),
        "smiles": [drugs[i] for i in rows],
        "sequence": [prots[j] for j in cols],
        "affinity": affinity[rows, cols].astype(np.float32),
    })

    n_train, n_test = EXPECTED_COUNTS[dataset]
    flat = [i for fold in train_folds for i in fold]
    assert len(train_folds) == N_FOLDS
    assert len(flat) == n_train and len(test_fold) == n_test, "fold sizes differ from the paper"
    assert len(df) == n_train + n_test, "pair count differs from the paper"
    assert len(set(flat) | set(test_fold)) == len(df), "folds overlap or miss pairs"
    return df, train_folds, test_fold


def check_ids(df):
    for key, value in (("drug_id", "smiles"), ("protein_id", "sequence")):
        if df.groupby(key)[value].nunique().max() != 1:
            raise ValueError(f"{key} does not identify a unique {value}")


def make_split(train_folds, test_fold, val_fold):
    """Validation = CV fold `val_fold`, train = the other 4 folds, test = the fixed test fold."""
    val = np.sort(np.asarray(train_folds[val_fold]))
    train = np.sort(np.concatenate(
        [np.asarray(f) for k, f in enumerate(train_folds) if k != val_fold]
    ))
    test = np.sort(np.asarray(test_fold))
    return train, val, test
