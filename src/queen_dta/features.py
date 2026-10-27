"""Feature engineering for QuEEN-DTA: fingerprints, token encodings, physicochemical
and quantum amplitude-embedding features."""
import numpy as np
import pennylane as qml
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.feature_extraction.text import CountVectorizer

from queen_dta.utils import DEVICE

QUEEN_VOCAB_SIZE = 41
QUEEN_PROTEIN_LEN = 500
QUEEN_SMILES_LEN = 100
QUEEN_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
QUEEN_STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
QUEEN_AA_TO_NUM = {aa: i + 1 for i, aa in enumerate(QUEEN_AMINO_ACIDS)}

dev_amp = qml.device("default.qubit", wires=2)
dev_simple = qml.device("default.qubit", wires=2)


@qml.qnode(dev_amp)
def biopython_amplitude_encoding(features):
    padded_features = np.pad(features, (0, 1), "constant")
    qml.AmplitudeEmbedding(
        padded_features, wires=range(2), normalize=True
    )
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]


@qml.qnode(dev_simple)
def q_feature_amp(x1, x2):
    features = np.array([x1, x2, 0.0, 0.0])
    qml.AmplitudeEmbedding(features, wires=range(2), normalize=True)
    return qml.expval(qml.PauliZ(0))


def apply_unique(function, values):
    values = np.asarray(values)
    if values.ndim == 1:
        unique_values, inverse = np.unique(values, return_inverse=True)
    else:
        unique_values, inverse = np.unique(
            values, axis=0, return_inverse=True
        )
    results = np.array([function(value) for value in unique_values])
    return results[np.asarray(inverse).reshape(-1)]


def single_biopython_features(seq):
    clean_seq = "".join(aa for aa in seq if aa in QUEEN_STANDARD_AA)
    if len(clean_seq) < 5:
        return np.zeros(3)
    analysed_seq = ProteinAnalysis(clean_seq)
    molar_ext = analysed_seq.molar_extinction_coefficient()
    return np.array(
        [molar_ext[0], molar_ext[1], analysed_seq.molecular_weight()]
    )


def get_ngrams(sequences, k=4):
    ngram_seqs = [
        " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))
        for seq in sequences
    ]
    vectorizer = CountVectorizer(max_features=512)
    return vectorizer.fit_transform(ngram_seqs).toarray()


def queen_encode_protein(sequences):
    encoded_list = []
    for seq in sequences:
        encoded = [QUEEN_AA_TO_NUM.get(aa, 0) for aa in seq]
        encoded = encoded[:QUEEN_PROTEIN_LEN]
        encoded += [0] * (QUEEN_PROTEIN_LEN - len(encoded))
        encoded_list.append(encoded)
    return np.array(encoded_list)


def protein_quantum_feature(seq):
    f1 = len(seq) / 1000.0
    f2 = len(set(seq)) / len(QUEEN_AMINO_ACIDS)
    return q_feature_amp(f1, f2)


def smiles_quantum_feature(smi):
    f1 = len(smi) / 100.0
    special = sum(smi.count(c) for c in ["=", "#", "(", ")"])
    f2 = special / 50.0
    return q_feature_amp(f1, f2)


def single_fingerprint(smi, morgan_gen):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.concatenate([np.zeros(167), np.zeros(512)])
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
    ecfp = np.array(morgan_gen.GetFingerprint(mol))
    return np.concatenate([maccs, ecfp])


def molecular_fingerprints(smiles):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=4, fpSize=512
    )
    return apply_unique(
        lambda smi: single_fingerprint(smi, morgan_gen), smiles
    )


def queen_encode_smiles(smiles, smi_dict):
    encoded = []
    for smi in smiles:
        tokens = [smi_dict.get(c, 0) for c in smi]
        tokens = tokens[:QUEEN_SMILES_LEN]
        tokens += [0] * (QUEEN_SMILES_LEN - len(tokens))
        encoded.append(tokens)
    return np.array(encoded)


def biopython_quantum(x_bp_scaled):
    x_bp_quantum = apply_unique(
        biopython_amplitude_encoding, x_bp_scaled
    )
    return np.concatenate([x_bp_scaled, x_bp_quantum], axis=1)


def prepare_queen(df):
    smiles = df["smiles"].values
    proteins = df["sequence"].values
    fingerprints = molecular_fingerprints(smiles)
    smi_chars = sorted(set("".join(smiles)))
    smi_dict = {c: i + 1 for i, c in enumerate(smi_chars)}
    x_smi_chars = queen_encode_smiles(smiles, smi_dict)
    x_smi_q = apply_unique(smiles_quantum_feature, smiles).reshape(-1, 1)
    x_smi = np.concatenate(
        [fingerprints, x_smi_chars, x_smi_q], axis=1
    ).astype(np.float32)
    x_bp = apply_unique(single_biopython_features, proteins)
    x_ngram = get_ngrams(proteins, k=4).astype(np.float32)
    x_prot_label = queen_encode_protein(proteins)
    x_prot_q = apply_unique(protein_quantum_feature, proteins)
    x_prot = (x_prot_label + x_prot_q.reshape(-1, 1)).astype(np.float32)
    return {
        "smi": x_smi,
        "prot": x_prot,
        "bp": x_bp.astype(np.float32),
        "ngram": x_ngram,
    }


def queen_inputs(features, idx, scalers):
    x_bp = scalers["bp"].transform(features["bp"][idx])
    return [
        scalers["smi"].transform(features["smi"][idx]),
        features["prot"][idx],
        biopython_quantum(x_bp),
        scalers["ngram"].transform(features["ngram"][idx]),
    ]


def queen_tensors(features, idx, scalers):
    smi, prot, bp, ngram = queen_inputs(features, idx, scalers)
    arrays = [
        np.ascontiguousarray(smi, dtype=np.float32),
        prot.astype(np.int32),
        np.ascontiguousarray(bp, dtype=np.float32),
        np.ascontiguousarray(ngram, dtype=np.float32),
    ]
    return [torch.from_numpy(a).to(DEVICE) for a in arrays]
