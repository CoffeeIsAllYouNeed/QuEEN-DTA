import numpy as np
import pandas as pd


class HandleDuplicates:

    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input Parameters:
            df (pd.DataFrame): Raw DataFrame.
        Output Parameters:
            df (pd.DataFrame): Deduplicated DataFrame.
        """
        try:
            df = df.drop_duplicates(
                subset=["drug_sequence", "protein_sequence"],
                keep="first",
            )
            df = df.dropna()
            df = df.reset_index(drop=True)
            return df
        except Exception as e:
            raise ValueError(f"Deduplication failed: {str(e)}")


class TypeConversion:

    def type_conversion(self, df: pd.DataFrame) -> tuple:
        """
        Input Parameters:
            df (pd.DataFrame): Deduplicated operational DataFrame.
        Output Parameters:
            tuple: Arrays of smiles, proteins, and float32 affinity scores.
        """
        try:
            smiles = (
                df["drug_sequence"].astype(str).values
            )
            proteins = (
                df["protein_sequence"].astype(str).values
            )
            scores = (
                df["affinity_score"].values.astype(np.float32)
            )
            return smiles, proteins, scores
        except Exception as e:
            raise TypeError(f"Type mapping failed: {str(e)}")


if __name__ == "__main__":
    pass