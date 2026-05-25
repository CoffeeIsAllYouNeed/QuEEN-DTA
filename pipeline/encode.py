import numpy as np
import pennylane as qml
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator


dev_amp = qml.device("default.qubit", wires=2)
dev_simple = qml.device("default.qubit", wires=2)


@qml.qnode(dev_amp)
def biopython_amplitude_encoding(features: np.ndarray) -> list:
    """
    Input Parameters:
        features (np.ndarray): Clean structural array of size 3.
    Output Parameters:
        list: Expected values following PauliZ quantum operations.
    """
    padded_features = np.pad(features, (0, 1), "constant")
    qml.AmplitudeEmbedding(
        padded_features, wires=range(2), normalize=True
    )
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]


@qml.qnode(dev_simple)
def q_feature_amp(x1: float, x2: float) -> float:
    """
    Input Parameters:
        x1 (float): First target analytical metric.
        x2 (float): Second target analytical metric.
    Output Parameters:
        float: Expected measurement outputs under PauliZ transformation.
    """
    features = np.array([x1, x2, 0.0, 0.0])
    qml.AmplitudeEmbedding(features, wires=range(2), normalize=True)
    return qml.expval(qml.PauliZ(0))


def encode_amplitude_biopython(
    X_biopython: np.ndarray,
) -> np.ndarray:
    """
    Input Parameters:
        X_biopython (np.ndarray): Input analytical array.
    Output Parameters:
        np.ndarray: Concatenated classic and quantum state arrays.
    """
    try:
        X_bp_quantum = np.array(
            [biopython_amplitude_encoding(f) for f in X_biopython]
        )
        return np.concatenate([X_biopython, X_bp_quantum], axis=1)
    except Exception as e:
        raise RuntimeError(
            f"Biopython Q-encoding dropped: {str(e)}"
        )


def encode_amplitude_protein(
    proteins: np.ndarray, AMINO_ACIDS: str
) -> np.ndarray:
    """
    Input Parameters:
        proteins (np.ndarray): Raw sequence array targets.
        AMINO_ACIDS (str): Complete standard token string matrix.
    Output Parameters:
        np.ndarray: Evaluated sequence array values.
    """
    try:
        res = []
        for seq in proteins:
            f1 = len(seq) / 1000.0
            f2 = len(set(seq)) / len(AMINO_ACIDS)
            res.append(q_feature_amp(f1, f2))
        return np.array(res).reshape(-1, 1)
    except Exception as e:
        raise RuntimeError(
            f"Protein Q-amplitude mapping error: {str(e)}"
        )


def encode_amplitude_smiles(smiles: np.ndarray) -> np.ndarray:
    """
    Input Parameters:
        smiles (np.ndarray): Target SMILES chemical string arrays.
    Output Parameters:
        np.ndarray: Tracked structural quantum projection arrays.
    """
    try:
        res = []
        for s in smiles:
            f1 = len(s) / 100.0
            special = sum(
                [s.count(c) for c in ["=", "#", "(", ")"]]
            )
            f2 = special / 50.0
            res.append(q_feature_amp(f1, f2))
        return np.array(res).reshape(-1, 1)
    except Exception as e:
        raise RuntimeError(
            f"SMILES Q-amplitude mapping error: {str(e)}"
        )


def encode_label_protein(
    proteins: np.ndarray, aa_to_num: dict
) -> np.ndarray:
    """
    Input Parameters:
        proteins (np.ndarray): Complete structural residue string target arrays.
        aa_to_num (dict): Map assigning integer levels to amino targets.
    Output Parameters:
        np.ndarray: Clean structural integer array configurations.
    """
    try:
        encoded_list = []
        for seq in proteins:
            encoded = [aa_to_num.get(aa, 0) for aa in seq]
            encoded = encoded[:500] + [0] * (500 - len(encoded))
            encoded_list.append(encoded)
        return np.array(encoded_list)
    except Exception as e:
        raise ValueError(
            f"Protein sequential mapping broke: {str(e)}"
        )


def encode_label_smiles(
    smiles: np.ndarray, smi_dict: dict
) -> np.ndarray:
    """
    Input Parameters:
        smiles (np.ndarray): Input operational dataset SMILES string arrays.
        smi_dict (dict): Maps structural components of character data.
    Output Parameters:
        np.ndarray: Vector array mapping elements to index keys.
    """
    try:
        encoded = []
        for s in smiles:
            tokens = [smi_dict.get(c, 0) for c in s]
            tokens = tokens[:100] + [0] * (100 - len(tokens))
            encoded.append(tokens)
        return np.array(encoded)
    except Exception as e:
        raise ValueError(
            f"SMILES structural encoding dropped: {str(e)}"
        )


def encode_molecular_fingerprints(
    smiles: np.ndarray,
) -> tuple:
    """
    Input Parameters:
        smiles (np.ndarray): Chemical identity formula string representations.
    Output Parameters:
        tuple: Matrix containing matching arrays of MACCS and ECFP states.
    """
    try:
        morgan_gen = (
            rdFingerprintGenerator.GetMorganGenerator(
                radius=4, fpSize=512
            )
        )
        maccs_fps = []
        ecfp_fps = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                maccs_fps.append(np.zeros(167))
                ecfp_fps.append(np.zeros(512))
            else:
                maccs_fps.append(
                    np.array(MACCSkeys.GenMACCSKeys(mol))
                )
                ecfp_fps.append(
                    np.array(morgan_gen.GetFingerprint(mol))
                )
        return np.array(maccs_fps), np.array(ecfp_fps)
    except Exception as e:
        raise RuntimeError(
            f"Molecular fingerprints uncalculated: {str(e)}"
        )


if __name__ == "__main__":
    pass