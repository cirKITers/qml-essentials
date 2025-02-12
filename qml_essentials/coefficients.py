from qml_essentials.model import Model
import numpy as np
from typing import Any


class Coefficients:

    @staticmethod
    def sample_coefficients(
        model: Model, shift=False, mfs: int = 1, mts: int = 1, **kwargs
    ) -> np.ndarray:
        """
        Extracts the coefficients of a given model using a FFT (np-fft).

        Note that the coefficients are complex numbers, but the imaginary part
        of the coefficients should be very close to zero, since the expectation
        values of the Pauli operators are real numbers.

        It can perform oversampling in both the frequency and time domain
        using the `mfs` and `mts` arguments.

        Args:
            model (Model): The model to sample.
            shift (bool): Whether to apply np-fftshift. Default is False.
            mfs (int): Multiplicator for the highest frequency. Default is 2.
            mts (int): Multiplicator for the number of time samples. Default is 1.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            np.ndarray: The sampled Fourier coefficients.
        """
        kwargs.setdefault("force_mean", True)
        kwargs.setdefault("execution_type", "expval")

        coeffs = Coefficients._fourier_transform(model, mfs=mfs, mts=mts, **kwargs)

        if not np.isclose(np.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is:\
                {np.sum(coeffs).imag}"
            )

        # Apply fftshift if required
        if shift:
            return np.fft.fftshift(coeffs)
        else:
            return coeffs

    @staticmethod
    def _fourier_transform(
        model: Model, mfs: int, mts: int, **kwargs: Any
    ) -> np.ndarray:
        # Create a frequency vector with as many frequencies as model degrees,
        # oversampled by nfs
        n_freqs: int = 2 * mfs * model.degree + 1

        # Create a vector of equally spaced time points
        nvecs = np.arange(-mfs * mts * model.degree, mfs * mts * model.degree + 1)

        # Stretch according to the number of frequencies
        inputs: np.ndarray = np.arange(0, mts * 2 * np.pi, 2 * np.pi / n_freqs)

        # Output vector is not necessarily the same length as input
        outputs: np.ndarray = np.zeros((mts * n_freqs))

        outputs = model(inputs=inputs, **kwargs)

        # Run the fft and rearrange + normalize the output
        return np.fft.fft(outputs) / outputs.size

    @staticmethod
    def get_frequencies(coeffs: np.ndarray, shift=False, mts=1) -> np.ndarray:
        """
        Get the frequencies corresponding to the given Fourier coefficients.

        Args:
            coeffs (np.ndarray): The Fourier coefficients.
            shift (bool): Whether to apply np-fftshift. Default is False.

        Returns:
            np.ndarray: The frequencies.
        """
        freqs = np.fft.fftfreq(coeffs.size, mts / coeffs.size)
        if shift:
            return np.fft.fftshift(freqs)
        else:
            return freqs
