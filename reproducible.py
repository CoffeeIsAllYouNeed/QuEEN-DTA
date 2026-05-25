import os
import random
import numpy as np
import tensorflow as tf


class Reproducible:
    """For strict determinism and GPU configurations."""

    def __init__(self, seed: int = 42) -> None:
        """
        Input Parameters:
            seed (int): The global random seed value.
        Output Parameters:
            None
        """
        self.seed = seed

    def set_determinism(self) -> None:
        """
        Input Parameters:
            None
        Output Parameters:
            None
        """
        try:
            os.environ["PYTHONHASHSEED"] = str(self.seed)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

            random.seed(self.seed)
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)
            tf.keras.utils.set_random_seed(self.seed)
            tf.config.experimental.enable_op_determinism()
        except Exception as e:
            raise RuntimeError(
                f"Error setting determinism: {str(e)}"
            )

    def initialize_gpu(self) -> tf.distribute.MirroredStrategy:
        """
        Input Parameters:
            None
        Output Parameters:
            strategy (tf.distribute.MirroredStrategy): GPU sync strategy.
        """
        try:
            physical_devices = tf.config.list_physical_devices("GPU")
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(
                        device, True
                    )
            strategy = tf.distribute.MirroredStrategy()
            return strategy
        except Exception as e:
            raise RuntimeError(f"GPU Init Failed: {str(e)}")


if __name__ == "__main__":
    rep = Reproducible()
    rep.set_determinism()
    strat = rep.initialize_gpu()