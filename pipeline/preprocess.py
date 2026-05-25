import pandas as pd
from pipeline.data_prep import (
    HandleDuplicates,
    TypeConversion,
)


def preprocess_pipeline(
    file_path: str,
) -> tuple:
    """
    Input Parameters:
        file_path (str): Target text file path string destination.
    Output Parameters:
        tuple: Arrays containing smiles, proteins, and float32 scores.
    """
    try:
        df = pd.read_csv(
            file_path, sep=" ", header=None
        )
        df.columns = [
            "drug_id",
            "protein_id",
            "drug_sequence",
            "protein_sequence",
            "affinity_score",
        ]

        dups_handler = HandleDuplicates()
        df = dups_handler.handle_duplicates(df)

        converter = TypeConversion()
        smiles, proteins, scores = (
            converter.type_conversion(df)
        )
        return smiles, proteins, scores
    except Exception as e:
        raise RuntimeError(
            f"Preprocessing failure: {str(e)}"
        )


if __name__ == "__main__":
    pass