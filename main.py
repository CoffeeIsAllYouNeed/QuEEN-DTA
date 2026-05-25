import os
import numpy as np
from pipeline.data_split import DataSplitter
from pipeline.evaluation import EvaluationMetric
from pipeline.feature_eng import FeatureEng
from pipeline.preprocess import preprocess_pipeline
from pipeline.scale import Scale
from pipeline.train_model import TrainModel
from reproducible import Reproducible

BATCH_SIZE = 64
EPOCHS = 415
SEED = 42


def run_pipeline(file_path: str) -> None:
    """
    Input Parameters:
        file_path (str): File path for target analysis.
    Output Parameters:
        None
    """
    if not os.path.exists(file_path):
        return

    # 1. Determinism Setup
    rep = Reproducible(seed=SEED)
    rep.set_determinism()
    strategy = rep.initialize_gpu()

    # 2. Loading and Preprocessing
    smiles, proteins, scores = preprocess_pipeline(
        file_path
    )

    # 3. Base Feature Engineering
    fe = FeatureEng()
    X_smi_final = fe.feature_extraction_smiles(smiles)
    X_bp_raw, X_ngrams, X_prot_label = (
        fe.feature_extraction_protein(proteins)
    )

    # 4. Sequential Split Distribution Protocol
    arrays_dict = {
        "smi": X_smi_final,
        "bp": X_bp_raw,
        "ngram": X_ngrams,
        "label": X_prot_label,
    }
    splitter = DataSplitter(seed=SEED)
    train, val, test = splitter.split_data(
        arrays_dict, scores
    )

    # 5. Feature Scaling Pipeline
    scaler = Scale()
    X_bp_tr, X_bp_v, X_bp_te = scaler.scale_biopython(
        train["bp"], val["bp"], test["bp"]
    )
    X_ng_tr, X_ng_v, X_ng_te = scaler.scale_ngrams(
        train["ngram"], val["ngram"], test["ngram"]
    )

    # 6. Post-Scale Quantum Features Integration
    X_bp_tr_q = fe.process_quantum_biopython(X_bp_tr)
    X_bp_v_q = fe.process_quantum_biopython(X_bp_v)
    X_bp_te_q = fe.process_quantum_biopython(X_bp_te)

    # 7. Quantum Sequential Alignment
    train_idx = int(len(train["y"]))
    val_idx = int(len(val["y"]))
    total_samples = len(scores)

    X_prot_tr_q = fe.process_quantum_protein(
        proteins[:train_idx], train["label"]
    )
    X_prot_v_q = fe.process_quantum_protein(
        proteins[train_idx : train_idx + val_idx],
        val["label"],
    )
    X_prot_te_q = fe.process_quantum_protein(
        proteins[train_idx + val_idx : total_samples],
        test["label"],
    )

    # 8. Model Assembly & Strategy Distribution Execution
    tm = TrainModel()
    with strategy.scope():
        model = tm.concatenate(
            smi_dim=train["smi"].shape[1],
            bp_dim=X_bp_tr_q.shape[1],
            ngram_dim=X_ng_tr.shape[1],
            vocab_size=41,
        )

    model.fit(
        [train["smi"], X_prot_tr_q, X_bp_tr_q, X_ng_tr],
        train["y"],
        validation_data=(
            [val["smi"], X_prot_v_q, X_bp_v_q, X_ng_v],
            val["y"],
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    # 9. Target Output Evaluation Generation
    curr_test_preds = model.predict(
        [test["smi"], X_prot_te_q, X_bp_te_q, X_ng_te],
        verbose=0,
    ).flatten()

    evaluator = EvaluationMetric()
    mse_val = evaluator.mean_squared_error(
        test["y"], curr_test_preds
    )
    r2m_val = evaluator.r2m(test["y"], curr_test_preds)
    ci_val = evaluator.concordance_index(
        test["y"], curr_test_preds
    )

    print(f"MSE: {mse_val:.3f}")
    print(f"R2M: {r2m_val:.3f}")
    print(f"CI: {ci_val:.3f}")


if __name__ == "__main__":
    davis_path = "data/davis.txt"
    kiba_path = "data/kiba.txt"

    run_pipeline(davis_path)
    run_pipeline(kiba_path)