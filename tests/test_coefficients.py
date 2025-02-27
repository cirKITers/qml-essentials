from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients, FourierTree
from pennylane.fourier import coefficients as pcoefficients

import numpy as np
import pennylane.numpy as pnp
import logging
import pytest

from functools import partial


logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_coefficients() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 3,
            "n_layers": 1,
            "output_qubit": [0, 1],
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_19",
            "n_qubits": 5,
            "n_layers": 1,
            "output_qubit": 0,
        },
    ]
    reference_inputs = np.linspace(-np.pi, np.pi, 10)

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            output_qubit=test_case["output_qubit"],
        )

        coeffs, _ = Coefficients.get_spectrum(model)

        assert len(coeffs) == model.degree * 2 + 1, "Wrong number of coefficients"
        assert np.isclose(
            np.sum(coeffs).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

        partial_circuit = partial(model, model.params)
        ref_coeffs = pcoefficients(partial_circuit, 1, model.degree)

        assert np.allclose(
            coeffs, ref_coeffs, rtol=1.0e-5
        ), "Coefficients don't match the pennylane reference"

        for ref_input in reference_inputs:
            exp_model = model(params=None, inputs=ref_input)

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=np.fft.fftshift(coeffs),
                input=ref_input,
            )

            assert np.isclose(
                exp_model, exp_fourier, atol=1.0e-5
            ), "Fourier series does not match model expectation"


@pytest.mark.unittest
def test_coefficients_tree() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 3,
            "n_layers": 1,
            "output_qubit": [0, 1],
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_19",
            "n_qubits": 3,
            "n_layers": 1,
            "output_qubit": 0,
        },
    ]

    reference_inputs = np.linspace(-np.pi, np.pi, 10)
    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            output_qubit=test_case["output_qubit"],
            as_pauli_circuit=False,
        )

        fft_coeffs, fft_freqs = Coefficients.get_spectrum(model, shift=True)

        coeff_tree = FourierTree(model)
        analytical_freqs, analytical_coeffs = coeff_tree(inputs=1.0).get_spectrum()

        assert len(analytical_freqs[0]) == len(
            analytical_freqs[0]
        ), "Wrong number of frequencies"
        assert np.isclose(
            np.sum(analytical_coeffs[0]).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

        if len(fft_coeffs) == len(analytical_coeffs[0]):
            assert all(np.isclose(fft_coeffs, analytical_coeffs[0], atol=1.0e-5)), (
                "FFT and analytical coefficients are not equal, despite same"
                "frequencies."
            )

        for ref_input in reference_inputs:
            exp_fourier_fft = Coefficients.evaluate_Fourier_series(
                coefficients=fft_coeffs, input=ref_input, frequencies=fft_freqs
            )

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=analytical_coeffs[0],
                input=ref_input,
                frequencies=analytical_freqs[0],
            )

            assert np.isclose(
                exp_fourier_fft, exp_fourier, atol=1.0e-5
            ), "FFT and analytical Fourier series do not match"


@pytest.mark.unittest
def test_oversampling_time() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    assert (
        Coefficients.get_spectrum(model, mts=2)[0].shape[0] == 10
    ), "Oversampling time failed"


@pytest.mark.unittest
def test_oversampling_frequency() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    assert (
        Coefficients.get_spectrum(model, mfs=2)[0].shape[0] == 9
    ), "Oversampling frequency failed"


@pytest.mark.unittest
def test_shift() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )
    coeffs, freqs = Coefficients.get_spectrum(model, shift=True)

    assert (
        np.abs(coeffs) == np.abs(coeffs[::-1])
    ).all(), "Shift failed. Spectrum must be symmetric."


@pytest.mark.smoketest
def test_frequencies() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )
    coeffs, freqs = Coefficients.get_spectrum(model, shift=True)

    assert (
        freqs.size == coeffs.size
    ), "Frequencies and coefficients must have the same length."
