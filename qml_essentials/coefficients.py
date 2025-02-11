from qml_essentials.model import Model
import numpy as np
from typing import Any


class Coefficients:

    @staticmethod
    def sample_coefficients(
        model: Model, shift=False, nfs: int = 2, nts: int = 1, **kwargs
    ) -> np.ndarray:
        """
        Sample the Fourier coefficients of a given model
        using Pennylane fourier.coefficients function.

        Note that the coefficients are complex numbers, but the imaginary part
        of the coefficients should be very close to zero, since the expectation
        values of the Pauli operators are real numbers.

        Args:
            model (Model): The model to sample.
            shift (bool): Whether to apply fftshift. Default is False.
            nfs (int): Multiplicator for the highest frequency. Default is 2.
            nts (int): Multiplicator for the number of time samples. Default is 1.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            np.ndarray: The sampled Fourier coefficients.
        """
        kwargs.setdefault("force_mean", True)
        kwargs.setdefault("execution_type", "expval")

        coeffs = Coefficients._fourier_transform(model, nfs=nfs, nts=nts, **kwargs)

        if not np.isclose(np.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is:\
                {np.sum(coeffs).imag}"
            )

        if shift:
            # Apply fftshift if required
            return np.fft.fftshift(coeffs)
        else:
            return coeffs

    @staticmethod
    def _fourier_transform(
        model: Model, nfs: int = 2, nts: int = 1, **kwargs: Any
    ) -> np.ndarray:
        """
        Perform a Fourier transform on the given model.

        Args:
            model (Model): The quantum model to transform.
            nfs (int): Number of frequency samples.
            nts (int): Number of time samples.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            np.ndarray: The Fourier-transformed data.
        """
        # Create a frequency vector with as many frequencies as model degrees,
        # oversampled by nfs
        n_freqs: int = int(nfs * model.degree + 1)

        # Create a vector of equally spaced time points
        nvecs = np.arange(-nts * model.degree, nts * model.degree + 1)

        # Stretch according to the number of frequencies
        inputs: np.ndarray = nvecs * (2 * np.pi / n_freqs)

        # Run the fft and rearrange + normalize the output
        return np.fft.fft(model(inputs=inputs, **kwargs)[nvecs - 1]) / inputs.size
