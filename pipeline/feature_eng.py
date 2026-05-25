import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from pipeline.encode import (
    encode_amplitude_biopython,
    encode_amplitude_protein,
    encode_amplitude_smiles,
    encode_label_protein,
    encode_label_smiles,
    encode_molecular_fingerprints,
)


class FeatureEng:

    def __init__(self) -> None:
        """
        Input Parameters:
            None
        Output Parameters:
            None
        """
        self.AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
        self.aa_to_num = {
            aa: idx + 1
            for idx, aa in enumerate(self.AMINO_ACIDS)
        }

    def extract_biopython_features(
        self, sequence_list: np.ndarray
    ) -> np.ndarray:
        """
        Input Parameters:
            sequence_list (np.ndarray): Set of targeted sequences.
        Output Parameters:
            np.ndarray: Calculated biochemical weight metrics.
        """
        try:
            features = []
            for seq in sequence_list:
                clean_seq = "".join(
                    [
                        aa
                        for aa in seq
                        if aa in "ACDEFGHIKLMNPQRSTVWY"
                    ]
                )
                if len(clean_seq) < 5:
                    features.append(np.zeros(3))
                    continue
                analysed_seq = ProteinAnalysis(clean_seq)
                molar_ext = (
                    analysed_seq.molar_extinction_coefficient()
                )
                ext_red = molar_ext[0]
                ext_ox = molar_ext[1]
                mw = analysed_seq.molecular_weight()
                features.append([ext_red, ext_ox, mw])
            return np.array(features)
        except Exception as e:
            raise RuntimeError(
                f"Biopython extraction failed: {str(e)}"
            )

    def get_ngrams(
        self, sequences: np.ndarray, k: int = 4
    ) -> np.ndarray:
        """
        Input Parameters:
            sequences (np.ndarray): Set of target sequences.
            k (int): Sliding window fragment length.
        Output Parameters:
            np.ndarray: Text fragment tracking metrics.
        """
        try:
            ngram_seqs = [
                " ".join(
                    [
                        seq[i : i + k]
                        for i in range(len(seq) - k + 1)
                    ]
                )
                for seq in sequences
            ]
            vectorizer = CountVectorizer(max_features=512)
            return vectorizer.fit_transform(
                ngram_seqs
            ).toarray()
        except Exception as e:
            raise ValueError(f"Ngram extraction dropped: {str(e)}")

    def feature_extraction_protein(
        self, proteins: np.ndarray
    ) -> tuple:
        """
        Input Parameters:
            proteins (np.ndarray): Target operational sequences.
        Output Parameters:
            tuple: Arrays of raw Biopython, raw N-grams, and labels.
        """
        try:
            X_bp_raw = self.extract_biopython_features(
                proteins
            )
            X_ngrams = self.get_ngrams(proteins, k=4)
            X_prot_label = encode_label_protein(
                proteins, self.aa_to_num
            )
            return X_bp_raw, X_ngrams, X_prot_label
        except Exception as e:
            raise RuntimeError(
                f"Protein downstream pipeline error: {str(e)}"
            )

    def feature_extraction_smiles(
        self, smiles: np.ndarray
    ) -> np.ndarray:
        """
        Input Parameters:
            smiles (np.ndarray): Input dataset formulas.
        Output Parameters:
            np.ndarray: Extracted array structure definitions.
        """
        try:
            X_maccs, X_ecfp = encode_molecular_fingerprints(
                smiles
            )
            smi_chars = sorted(
                list(set("".join(smiles)))
            )
            smi_dict = {
                c: i + 1 for i, c in enumerate(smi_chars)
            }
            X_smi_chars = encode_label_smiles(
                smiles, smi_dict
            )
            X_smi_q = encode_amplitude_smiles(smiles)
            return np.concatenate(
                [X_maccs, X_ecfp, X_smi_chars, X_smi_q],
                axis=1,
            )
        except Exception as e:
            raise RuntimeError(
                f"SMILES downstream pipeline error: {str(e)}"
            )

    def process_quantum_protein(
        self, proteins: np.ndarray, X_prot_label: np.ndarray
    ) -> np.ndarray:
        """
        Input Parameters:
            proteins (np.ndarray): Input sequence identities.
            X_prot_label (np.ndarray): Mapped base character matrix.
        Output Parameters:
            np.ndarray: Enhanced quantum features array.
        """
        try:
            X_prot_q = encode_amplitude_protein(
                proteins, self.AMINO_ACIDS
            )
            X_prot_q_expanded = np.repeat(
                X_prot_q, X_prot_label.shape[1], axis=1
            )
            return X_prot_label + X_prot_q_expanded
        except Exception as e:
            raise RuntimeError(
                f"Quantum sequence blending dropped: {str(e)}"
            )

    def process_quantum_biopython(
        self, X_bp_scaled: np.ndarray
    ) -> np.ndarray:
        """
        Input Parameters:
            X_bp_scaled (np.ndarray): Scaled classic features array.
        Output Parameters:
            np.ndarray: Combined quantum-classic arrays.
        """
        try:
            return encode_amplitude_biopython(X_bp_scaled)
        except Exception as e:
            raise RuntimeError(
                f"Quantum Biopython blending dropped: {str(e)}"
            )


if __name__ == "__main__":
    pass
