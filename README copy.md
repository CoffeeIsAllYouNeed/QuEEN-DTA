# QuEEN-DTA on the DeepH-DTA protocol

Runs QuEEN-DTA with the data, split and training schedule of
**DeepH-DTA** (Abdel-Basset et al., IEEE Access 8, 2020, doi:10.1109/ACCESS.2020.3024238),
so the result can be set next to the 13 baselines in that paper's Tables 3 and 4.

## Run

1. Open `run_queen_dta.ipynb` from this folder (Python 3.10+, internet for the first install).
2. Run the cells in order. Cell 1 creates `.venv` inside this folder and installs everything
   there (PyTorch CUDA 12.6 build plus the packages listed at the top of that cell).
3. Cell 3 trains. Finished runs are saved in `results/` and skipped on a re-run, so the notebook
   can be stopped and started again.

## What matches DeepH-DTA

| item | value |
|---|---|
| data | DeepDTA Davis (30,056 pairs) and KIBA (118,254 pairs), stored in `data/` |
| SMILES | RDKit canonical isomeric SMILES, as in `reference/deeph_dta_create_data.py` |
| Davis label | pKd = -log10(Kd / 1e9); KIBA score unchanged |
| test set | the DeepDTA test fold, fixed: Davis 5,010 pairs, KIBA 19,709 pairs |
| train / validation | the 5 DeepDTA training folds; per run, 1 fold is validation and the other 4 are train (Davis about 20,037 / 5,009, KIBA 78,836 / 19,709, as in the paper's Table 2) |
| runs | 5 per dataset (each fold is validation once), mean and std reported |
| epochs | 200 |
| metrics | CI, MSE, r_m^2 (R^2 is reported in addition) |

Davis fold 0 holds 5,010 pairs, so the run with fold 0 as validation has 20,036 train / 5,010 validation.

## What DeepH-DTA does not state (set here, change in `queen_dta/protocol.py`)

- **Random seed.** The paper and the released code contain none, and the released code has no
  training script. Run k uses seed `SEED_BASE + k` (0 to 4).
- **Which weights are tested.** Default: the epoch with the best validation MSE
  (`SELECT_BEST_VALIDATION = True`). Set it to `False` for the last epoch.
- **Optimiser settings.** Default keeps QuEEN-DTA's tuned values (batch 64, lr 0.000131,
  dropout 0.29, Huber loss, weight decay 0.004). `MATCH_REFERENCE_OPTIMISER = True` switches to
  DeepH-DTA's Table 1 values (batch 256, Adam lr 1e-5, dropout 0.1, MSE loss).
- **AUPC** (reported by DeepH-DTA) is not computed.

## Comparison

`reference/deeph_dta_published.json` holds the numbers printed in DeepH-DTA's Tables 3 and 4
(KronRLS, SimBoost, DeepDTA, MT-DTI (wo-FT), MT-DTI, DeepCPI, WideDTA, GANsDTA, AttentionDTA,
GAT, GIN [11], GIN [12], DeepGS, DeepH-DTA). Those baselines were not re-run here. The last cell
lists the models QuEEN-DTA beats on CI, MSE and r_m^2 on both datasets, with QuEEN-DTA rounded to 3
decimals and a strict comparison.

## Output

`results/<dataset>_fold<k>.json|npz` per run, `scores_per_fold.csv`, `scores_summary.csv`,
`comparison_with_deeph_dta_tables.csv`.

## Layout

```
run_queen_dta.ipynb      launcher (venv, GPU check, training, report)
run_experiment.py        training entry point (runs inside .venv)
report.py                summary and comparison
queen_dta/protocol.py    all protocol settings
queen_dta/data.py        data and split from the DeepDTA files
queen_dta/model.py       QuEEN-DTA features and network
queen_dta/train.py       training loop, one run per fold
data/                    Davis and KIBA with the DeepDTA folds
reference/               DeepH-DTA data code and the published table values
```
