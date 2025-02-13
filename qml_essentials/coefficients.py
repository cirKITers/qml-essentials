from qml_essentials.model import Model
import numpy as np
from typing import Any


class Coefficients:

    @staticmethod
    def get_spectrum(
        model: Model, mfs: int = 1, mts: int = 1, shift=False, trim=False, **kwargs
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
            mfs (int): Multiplicator for the highest frequency. Default is 2.
            mts (int): Multiplicator for the number of time samples. Default is 1.
            shift (bool): Whether to apply np-fftshift. Default is False.
            trim (bool): Whether to remove the Nyquist frequency if spectrum is even.
                Default is False.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            np.ndarray: The sampled Fourier coefficients.
        """
        kwargs.setdefault("force_mean", True)
        kwargs.setdefault("execution_type", "expval")

        coeffs, freqs = Coefficients._fourier_transform(
            model, mfs=mfs, mts=mts, **kwargs
        )

        if not np.isclose(np.sum(coeffs).imag, 0.0, rtol=1.0e-5):
            raise ValueError(
                f"Spectrum is not real. Imaginary part of coefficients is:\
                {np.sum(coeffs).imag}"
            )

        if trim and coeffs.size % 2 == 0:
            coeffs = np.delete(coeffs, len(coeffs) // 2)
            freqs = np.delete(freqs, len(freqs) // 2)

        if shift:
            return np.fft.fftshift(coeffs), np.fft.fftshift(freqs)
        else:
            return coeffs, freqs

    @staticmethod
    def _fourier_transform(
        model: Model, mfs: int, mts: int, **kwargs: Any
    ) -> np.ndarray:
        # Create a frequency vector with as many frequencies as model degrees,
        # oversampled by nfs
        n_freqs: int = 2 * mfs * model.degree + 1

        # Stretch according to the number of frequencies
        inputs: np.ndarray = np.arange(0, 2 * mts * np.pi, 2 * np.pi / n_freqs)

        # Output vector is not necessarily the same length as input
        outputs: np.ndarray = np.zeros((mts * n_freqs))

        outputs = model(inputs=inputs, **kwargs)

        coeffs = np.fft.fft(outputs)

        freqs = np.fft.fftfreq(coeffs.size, mts / coeffs.size)

        # Run the fft and rearrange + normalize the output
        return coeffs / outputs.size, freqs

    @staticmethod
    def get_psd(coeffs: np.ndarray) -> np.ndarray:
        # TODO: if we apply trim=True in advance, this will be slightly wrong..
        def abs2(x):
            return x.real**2 + x.imag**2

        scale = 2.0 / (len(coeffs) ** 2)
        return scale * abs2(coeffs)
