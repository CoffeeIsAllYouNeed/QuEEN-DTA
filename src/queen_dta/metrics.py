"""Evaluation metrics: MSE, concordance index (CI), r_m^2 and R^2."""
import numpy as np
import torch
from sklearn.metrics import r2_score

from queen_dta.utils import DEVICE


def mse(y, f):
    return float(np.mean((y - f) ** 2))


def concordance_index(y, f, chunk=2048):
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=DEVICE)
    f = torch.as_tensor(np.asarray(f), dtype=torch.float32, device=DEVICE)
    comparable = torch.zeros((), dtype=torch.int64, device=DEVICE)
    better = torch.zeros((), dtype=torch.int64, device=DEVICE)
    ties = torch.zeros((), dtype=torch.int64, device=DEVICE)
    for start in range(0, len(y), chunk):
        mask = y[start:start + chunk, None] > y[None, :]
        diff = f[start:start + chunk, None] - f[None, :]
        comparable += mask.sum()
        better += ((diff > 0) & mask).sum()
        ties += ((diff == 0) & mask).sum()
    total = int(comparable)
    if total == 0:
        return 0.0
    return (int(better) + 0.5 * int(ties)) / total


def r2m(y, f):
    r2 = np.corrcoef(y, f)[0, 1] ** 2
    k = np.sum(y * f) / np.sum(f * f)
    r02 = 1 - np.sum((y - k * f) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2 * (1 - np.sqrt(np.abs(r2 * r2 - r02 * r02))))


def evaluate(y, f):
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    return {
        "mse": mse(y, f),
        "ci": concordance_index(y, f),
        "r2m": r2m(y, f),
        "r2": float(r2_score(y, f)),
    }
