from qml_essentials.model import Model
from functools import partial
from pennylane.fourier import coefficients
import numpy as np
from typing import Optional


class Coefficients:

    @staticmethod
    def sample_coefficients(model: Model, **kwargs) -> np.ndarray:
        """
        Sample the Fourier coefficients of a given model
        using Pennylane fourier.coefficients function.

        Note that the coefficients are complex numbers, but the imaginary part
        of the coefficients should be very close to zero, since the expectation
        values of the Pauli operators are real numbers.

        Args:
            model (Model): The model to sample.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            np.ndarray: The sampled Fourier coefficients.
        """
        kwargs.setdefault("force_mean", True)
        kwargs.setdefault("execution_type", "expval")

        partial_circuit = partial(model, model.params, **kwargs)
        coeffs = coefficients(partial_circuit, 1, model.degree)

        if not np.isclose(np.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is:\
                {np.sum(coeffs).imag}"
            )

        return coeffs

    @staticmethod
    def evaluate_Fourier_series(
        coefficients: np.ndarray,
        input: float,
        frequencies: Optional[np.ndarray] = None,
    ) -> float:
        """
        Evaluate the function value of a Fourier series at one point.

        Args:
            coefficients (np.ndarray): Coefficients of the Fourier series.
            input (float): Point at which to evaluate the function.
            frequencies (Optional[np.ndarray]): Corresponding frequencies in
                the form [-n_freq, ..., 0, ..., n_freq]. If None, the number of
                coefficients is to obtain sequential frequencies.

        Returns:
            float: The function value at the input point.
        """
        n_freq = len(coefficients) // 2
        if frequencies is None:
            frequencies = np.arange(-n_freq, n_freq + 1)
        pos_coeff = coefficients[n_freq + 1 :]
        neg_coeff = coefficients[:n_freq][::-1]

        assert all(np.isclose(np.conjugate(pos_coeff), neg_coeff, atol=1e-5)), (
            "Coefficients for negative frequencies should be the complex "
            "conjugate of the respective positive ones."
        )

        exp = coefficients[n_freq]  # zero coefficient
        for omega in range(1, n_freq + 1):
            exp += pos_coeff[omega - 1] * np.exp(1j * omega * input)
            exp += neg_coeff[omega - 1] * np.exp(-1j * omega * input)
        return exp
