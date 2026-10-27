# Data

Davis and KIBA exactly as released with DeepDTA (Ozturk et al., 2018,
https://github.com/hkmztrk/DeepDTA, folder `data/`). These are the files that
the benchmark protocol uses.

Per dataset:

- `Y`                              affinity matrix (drugs x proteins), NaN = not measured
- `ligands_can.txt`                drug SMILES
- `proteins.txt`                   protein sequences
- `folds/train_fold_setting1.txt`  5 lists of pair positions (the 5 cross-validation folds)
- `folds/test_fold_setting1.txt`   1 list of pair positions (the fixed test set)

A pair position is an index into `np.where(~np.isnan(Y))`, flattened row by row.
Davis affinities are converted to pKd with `-log10(Kd / 1e9)`, KIBA scores are used as they are.

| dataset | pairs   | train folds (5)           | test   |
|---------|---------|---------------------------|--------|
| Davis   | 30,056  | 5,010 + 4 x 5,009 = 25,046 | 5,010  |
| KIBA    | 118,254 | 5 x 19,709 = 98,545        | 19,709 |
