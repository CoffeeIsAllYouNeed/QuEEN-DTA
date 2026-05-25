import numpy as np


class DataSplitter:
    """8:1:1 split strategy for train, validation, and test sets."""

    def __init__(self, seed: int = 42) -> None:
        """
        Input Parameters:
            seed (int): Set to 42 as it is the default global random seed value.
        Output Parameters:
            None
        """
        self.seed = seed

    def split_data(
        self, arrays_dict: dict, y: np.ndarray
    ) -> tuple:
        """
        Input Parameters:
            arrays_dict (dict): Maps raw tracking tags to arrays.
            y (np.ndarray): Shared scalar evaluation targets.
        Output Parameters:
            tuple: Structured train_data, validation_data, and test_data.
        """
        try:
            total_samples = len(y)
            indices = np.arange(total_samples)

            np.random.seed(self.seed)
            np.random.shuffle(indices)

            test_size = int(0.10 * total_samples)
            val_size = int(0.10 * total_samples)
            train_size = (
                total_samples - test_size - val_size
            )

            train_idx = indices[:train_size]
            val_idx = indices[
                train_size : train_size + val_size
            ]
            test_idx = indices[train_size + val_size :]

            train_data = {}
            validation_data = {}
            test_data = {}

            for k, arr in arrays_dict.items():
                train_data[k] = arr[train_idx]
                validation_data[k] = arr[val_idx]
                test_data[k] = arr[test_idx]

            train_data["y"] = y[train_idx]
            validation_data["y"] = y[val_idx]
            test_data["y"] = y[test_idx]

            return train_data, validation_data, test_data
        except Exception as e:
            raise ValueError(f"Splitting protocol failed: {str(e)}")


if __name__ == "__main__":
    pass