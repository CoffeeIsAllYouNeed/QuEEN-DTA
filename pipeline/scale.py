import numpy as np
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


class Scale:

    def __init__(self) -> None:
        """
        Input Parameters:
            None
        Output Parameters:
            None
        """
        self.bp_scaler = StandardScaler()
        self.ngram_scaler = MaxAbsScaler()
        self.smi_scaler = MaxAbsScaler()

    def scale_biopython(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> tuple:
        """
        Input Parameters:
            X_train (np.ndarray): Raw training Biopython array.
            X_val (np.ndarray): Raw validation Biopython array.
            X_test (np.ndarray): Raw test Biopython array.
        Output Parameters:
            tuple: Standardized split distributions.
        """
        try:
            X_tr = self.bp_scaler.fit_transform(X_train)
            X_v = self.bp_scaler.transform(X_val)
            X_te = self.bp_scaler.transform(X_test)
            return X_tr, X_v, X_te
        except Exception as e:
            raise ValueError(
                f"Biopython scaling failed: {str(e)}"
            )

    def scale_ngrams(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> tuple:
        """
        Input Parameters:
            X_train (np.ndarray): Raw training N-gram array.
            X_val (np.ndarray): Raw validation N-gram array.
            X_test (np.ndarray): Raw test N-gram array.
        Output Parameters:
            tuple: Absolute-scaled split distributions.
        """
        try:
            X_tr = self.ngram_scaler.fit_transform(X_train)
            X_v = self.ngram_scaler.transform(X_val)
            X_te = self.ngram_scaler.transform(X_test)
            return X_tr, X_v, X_te
        except Exception as e:
            raise ValueError(f"Ngram scaling failed: {str(e)}")

    def scale_smiles(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> tuple:
        """
        Input Parameters:
            X_train (np.ndarray): Raw training SMILES array.
            X_val (np.ndarray): Raw validation SMILES array.
            X_test (np.ndarray): Raw test SMILES array.
        Output Parameters:
            tuple: Absolute-scaled split distributions.
        """
        try:
            X_tr = self.smi_scaler.fit_transform(X_train)
            X_v = self.smi_scaler.transform(X_val)
            X_te = self.smi_scaler.transform(X_test)
            return X_tr, X_v, X_te
        except Exception as e:
            raise ValueError(f"SMILES scaling failed: {str(e)}")


if __name__ == "__main__":
    pass