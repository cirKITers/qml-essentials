from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients

import numpy as np
import pennylane.numpy as pnp
import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_coefficients() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 1,
            "n_layers": 3,
            "n_samples": 200,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "n_samples": 200,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "n_samples": 200,
            "output_qubit": [0, 1, 2, 3],
        },
    ]
    reference_inputs = np.linspace(-np.pi, np.pi, 10)

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=True,
            initialization="random",
            output_qubit=test_case["output_qubit"],
        )

        coeffs = Coefficients.sample_coefficients(model)

        assert (
            len(coeffs) == model.degree * 2 + 1
        ), "Wrong number of coefficients"
        assert np.isclose(
            np.sum(coeffs).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

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
            "circuit_type": "Circuit_19",
            "n_qubits": 1,
            "n_layers": 3,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "output_qubit": [0, 1, 2, 3],
        },
    ]

    reference_inputs = np.linspace(-np.pi, np.pi, 10)
    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=True,
            initialization="random",
            output_qubit=test_case["output_qubit"],
            as_pauli_circuit=True,
        )

        fft_coeffs = Coefficients.sample_coefficients(model)
        fft_coeffs = np.fft.fftshift(fft_coeffs)

        coeff_tree = model.build_coefficients_tree(
            pnp.tensor(model.params),
            inputs=None,
            force_mean=True,
            execution_type="expval",
        )
        analytical_freqs, analytical_coeffs = coeff_tree.get_spectrum()

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
                coefficients=fft_coeffs,
                input=ref_input,
            )

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=analytical_coeffs[0],
                input=ref_input,
            )

            assert np.isclose(
                exp_fourier_fft, exp_fourier, atol=1.0e-5
            ), "Fourier series does not match model expectation"
