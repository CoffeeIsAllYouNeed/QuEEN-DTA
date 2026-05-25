import numpy as np
from math import sqrt
from sklearn.metrics import r2_score


class EvaluationMetric:
    """Metrics as per standard research papers."""

    def mean_squared_error(
        self, y: np.ndarray, f: np.ndarray
    ) -> float:
        """
        Input Parameters:
            y (np.ndarray): Exact true continuous scalar distributions.
            f (np.ndarray): Output regression prediction values.
        Output Parameters:
            float: Evaluated Mean Squared Error value.
        """
        try:
            return float(((y - f) ** 2).mean(axis=0))
        except Exception as e:
            raise ArithmeticError(
                f"MSE evaluation failure: {str(e)}"
            )

    def r2m(self, y: np.ndarray, f: np.ndarray) -> float:
        """
        Input Parameters:
            y (np.ndarray): Target true metric vector sets.
            f (np.ndarray): Derived target model tracking outputs.
        Output Parameters:
            float: Evaluated modified R2 spatial metrics value.
        """
        try:
            r2 = r2_score(y, f)
            r02 = r2_score(y, f, force_finite=False)
            return float(
                r2 * (1 - np.sqrt(np.abs(r2 - r02)))
            )
        except Exception as e:
            raise ArithmeticError(
                f"R2M evaluation failure: {str(e)}"
            )

    def concordance_index(
        self, y: np.ndarray, f: np.ndarray
    ) -> float:
        """
        Input Parameters:
            y (np.ndarray): True target values.
            f (np.ndarray): Matched computed model predictions.
        Output Parameters:
            float: Evaluated concordance probability index value.
        """
        try:
            ind = np.argsort(y)
            y = y[ind]
            f = f[ind]
            i = len(y) - 1
            j = i - 1
            z = 0.0
            S = 0.0
            while i > 0:
                while j >= 0:
                    if y[i] > y[j]:
                        z = z + 1
                        u = f[i] - f[j]
                        if u > 0:
                            S = S + 1
                        elif u == 0:
                            S = S + 0.5
                    j = j - 1
                i = i - 1
                j = i - 1
            if z == 0:
                return 0.0
            return float(S / z)
        except Exception as e:
            raise ArithmeticError(
                f"CI evaluation failure: {str(e)}"
            )


if __name__ == "__main__":
    pass