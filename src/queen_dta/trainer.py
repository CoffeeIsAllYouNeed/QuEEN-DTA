"""Training of QuEEN-DTA, one run per cross-validation fold."""
import functools
import json
import traceback

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from queen_dta import config as P
from queen_dta.dataset import check_ids, load_dataset, make_split
from queen_dta.features import prepare_queen, queen_tensors
from queen_dta.metrics import evaluate
from queen_dta.model import QueenNet, queen_forward
from queen_dta.optim import KerasAdam
from queen_dta.utils import DEVICE, release_memory, set_seed


def pack(val_idx, val_pred, test_idx, test_pred):
    return {
        "val_idx": np.asarray(val_idx),
        "val_pred": np.asarray(val_pred, dtype=np.float64).reshape(-1),
        "test_idx": np.asarray(test_idx),
        "test_pred": np.asarray(test_pred, dtype=np.float64).reshape(-1),
    }


def torch_predict(model, forward_fn, rows, batch_size=1024):
    model.eval()
    rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size].to(DEVICE)
            outputs.append(forward_fn(chunk).reshape(-1).cpu())
    return torch.cat(outputs).numpy()


def result_path(dataset, fold, suffix):
    return P.RESULTS_DIR / f"{dataset}_fold{fold}.{suffix}"


def predict_queen(label, features, df, split, test_idx, seed):
    train_idx, val_idx = split
    scores = df["affinity"].values.astype(np.float32)
    scalers = {
        "smi": MaxAbsScaler().fit(features["smi"][train_idx]),
        "bp": StandardScaler().fit(features["bp"][train_idx]),
        "ngram": MaxAbsScaler().fit(features["ngram"][train_idx]),
    }
    train_t = queen_tensors(features, train_idx, scalers)
    val_t = queen_tensors(features, val_idx, scalers)
    test_t = queen_tensors(features, test_idx, scalers)
    y_train = torch.from_numpy(scores[train_idx]).to(DEVICE)
    y_val = scores[val_idx].astype(np.float64)

    set_seed(seed)
    model = QueenNet(
        train_t[0].shape[1], train_t[2].shape[1], train_t[3].shape[1]
    ).to(DEVICE)
    optimizer = KerasAdam(
        model.parameters(),
        lr=P.QUEEN_LEARNING_RATE,
        eps=1e-7,
        weight_decay=P.QUEEN_WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss() if P.QUEEN_LOSS == "mse" else nn.HuberLoss(delta=1.0)
    generator = torch.Generator().manual_seed(seed)
    n_train = len(train_idx)
    val_forward = functools.partial(queen_forward, model, val_t)
    val_rows = np.arange(len(val_idx))

    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    waiting = 0
    epochs_run = 0
    for epoch in range(P.MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(DEVICE)
        for start in range(0, n_train, P.QUEEN_BATCH_SIZE):
            batch = order[start:start + P.QUEEN_BATCH_SIZE]
            if len(batch) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            out = queen_forward(model, train_t, batch).view(-1)
            loss = loss_fn(out, y_train[batch])
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1

        val_now = torch_predict(model, val_forward, val_rows)
        val_mse = float(np.mean((y_val - val_now) ** 2))
        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epochs_run
            waiting = 0
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        else:
            waiting += 1
        if epochs_run % 10 == 0:
            print(
                f"{label} epoch {epochs_run} | val MSE {val_mse:.4f} "
                f"| best {best_mse:.4f} (epoch {best_epoch})",
                flush=True,
            )
        if P.PATIENCE is not None and waiting > P.PATIENCE:
            print(f"{label} stopped at epoch {epochs_run}", flush=True)
            break

    if P.SELECT_BEST_VALIDATION:
        model.load_state_dict(best_state)
    val_pred = torch_predict(model, val_forward, val_rows)
    test_pred = torch_predict(
        model,
        functools.partial(queen_forward, model, test_t),
        np.arange(len(test_idx)),
    )
    out = pack(val_idx, val_pred, test_idx, test_pred)
    out["epochs_run"] = np.asarray(epochs_run)
    out["best_epoch"] = np.asarray(best_epoch)
    return out


def save_result(dataset, fold, seed, out, y, n_train):
    record = {
        "dataset": dataset,
        "model": "QuEEN-DTA",
        "val_fold": int(fold),
        "seed": int(seed),
        "n_train": int(n_train),
        "n_val": len(out["val_idx"]),
        "n_test": len(out["test_idx"]),
        "epochs_run": int(out["epochs_run"]),
        "best_epoch": int(out["best_epoch"]),
        "weights": "best validation MSE" if P.SELECT_BEST_VALIDATION else "last epoch",
    }
    for part in ("val", "test"):
        metrics = evaluate(y[out[f"{part}_idx"]], out[f"{part}_pred"])
        record.update({f"{part}_{k}": v for k, v in metrics.items()})
    np.savez(result_path(dataset, fold, "npz"), **out)
    result_path(dataset, fold, "json").write_text(json.dumps(record))
    print(
        f"{dataset} | fold {fold} | best epoch {record['best_epoch']} of "
        f"{record['epochs_run']} | val MSE {record['val_mse']:.4f} | "
        f"test MSE {record['test_mse']:.4f} | test CI {record['test_ci']:.4f} | "
        f"test R2M {record['test_r2m']:.4f}",
        flush=True,
    )


def run_dataset(dataset, folds):
    P.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pending = [f for f in folds if not result_path(dataset, f, "json").exists()]
    if not pending:
        print(f"{dataset}: all requested folds already saved", flush=True)
        return
    features = None
    try:
        df, train_folds, test_fold = load_dataset(dataset)
        check_ids(df)
        print(f"{dataset}: {len(df):,} pairs, building features ...", flush=True)
        features = prepare_queen(df)
        for fold in pending:
            train_idx, val_idx, test_idx = make_split(train_folds, test_fold, fold)
            print(
                f"{dataset} | validation fold {fold} | train {len(train_idx):,} | "
                f"validation {len(val_idx):,} | test {len(test_idx):,}",
                flush=True,
            )
            seed = P.SEED_BASE + fold
            out = predict_queen(
                f"{dataset} fold {fold}", features, df,
                (train_idx, val_idx), test_idx, seed,
            )
            save_result(dataset, fold, seed, out, df["affinity"].values, len(train_idx))
            release_memory()
    except Exception as error:
        print(f"{dataset} stopped: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
    finally:
        del features
        release_memory()
