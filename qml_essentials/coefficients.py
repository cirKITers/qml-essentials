from qml_essentials.model import Model
from functools import partial
from pennylane.fourier import coefficients
import numpy as np


class Coefficients:

    @staticmethod
    def sample_coefficients(model: Model) -> np.ndarray:
        """
        Sample the Fourier coefficients of a given model
        using Pennylane fourier.coefficients function.

        Note that the coefficients are complex numbers, but the imaginary part
        of the coefficients should be very close to zero, since the expectation
        values of the Pauli operators are real numbers.

        Args:
            model (Model): The model to sample.

        Returns:
            np.ndarray: The sampled Fourier coefficients.
        """
        partial_circuit = partial(model, model.params)
        coeffs = coefficients(partial_circuit, 1, model.degree)

        if not np.isclose(np.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is: {np.sum(coeffs).imag}"
            )

        return coeffs
