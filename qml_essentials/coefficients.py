from qml_essentials.model import Model
from functools import partial
from pennylane.fourier import coefficients
import numpy as np


class Coefficients:

    def sample_coefficients(model: Model) -> np.ndarray:
        """
        Sample the Fourier coefficients of a given model.

        Args:
            model (Model): The model to sample.

        Returns:
            np.ndarray: The sampled Fourier coefficients.
        """
        partial_circuit = partial(model, model.params)
        return coefficients(partial_circuit, 1, model.degree)
