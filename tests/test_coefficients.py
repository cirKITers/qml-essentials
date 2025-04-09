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

        coeffs, freqs = Coefficients.get_spectrum(model)

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
                coefficients=coeffs,
                frequencies=freqs,
                inputs=ref_input,
            )

            assert np.isclose(
                exp_model, exp_fourier, atol=1.0e-5
            ), "Fourier series does not match model expectation"


@pytest.mark.unittest
def test_multi_dim_input() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        output_qubit=-1,
        encoding=["RX", "RX"],
    )

    coeffs, freqs = Coefficients.get_spectrum(model)

    assert (
        coeffs.shape == (model.degree * 2 + 1,) * model.n_input_feat
    ), f"Wrong shape of coefficients: {coeffs.shape}, \
        expected {(model.degree*2+1,)*model.n_input_feat}"

    ref_input = [1, 2]
    exp_model = model(params=None, inputs=ref_input, force_mean=True)
    exp_fourier = Coefficients.evaluate_Fourier_series(
        coefficients=coeffs,
        frequencies=freqs,
        inputs=ref_input,
    )

    assert np.isclose(
        exp_model, exp_fourier, atol=1.0e-5
    ), "Fourier series does not match model expectation"


@pytest.mark.smoketest
def test_batch() -> None:
    n_samples = 3

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_15",
        output_qubit=-1,
        mp_threshold=100,
        initialization="random",
    )

    model.initialize_params(rng=pnp.random.default_rng(1000), repeat=n_samples)
    params = model.params
    coeffs_parallel, _ = Coefficients.get_spectrum(model, shift=True)

    # TODO: once the code is ready, test frequency vector as well
    for i in range(n_samples):
        model.params = params[:, :, i]
        coeffs_single, _ = Coefficients.get_spectrum(model, shift=True)
        assert np.allclose(
            coeffs_parallel[:, i], coeffs_single, rtol=1.0e-5
        ), "MP and SP coefficients don't match for 1D input"

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        output_qubit=-1,
        mp_threshold=100,
        encoding=["RX", "RY"],
        initialization="random",
    )

    model.initialize_params(rng=pnp.random.default_rng(1000), repeat=n_samples)
    params = model.params
    coeffs_parallel, _ = Coefficients.get_spectrum(model, shift=True)

    for i in range(n_samples):
        model.params = params[:, :, i]
        coeffs_single, _ = Coefficients.get_spectrum(model, shift=True)
        assert np.allclose(
            coeffs_parallel[:, :, i], coeffs_single, rtol=1.0e-5
        ), "MP and SP coefficients don't match for 2D input"


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
        analytical_coeffs, analytical_freqs = coeff_tree.get_spectrum(force_mean=True)

        assert len(analytical_freqs[0]) == len(
            analytical_freqs[0]
        ), "Wrong number of frequencies"
        assert np.isclose(
            np.sum(analytical_coeffs[0]).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

        # Filter fft_coeffs for only the frequencies that occur in the spectrum
        sel_fft_coeffs = np.take(fft_coeffs, analytical_freqs[0] + int(max(fft_freqs)))
        assert all(
            np.isclose(sel_fft_coeffs, analytical_coeffs[0], atol=1.0e-5)
        ), "FFT and analytical coefficients are not equal."

        for ref_input in reference_inputs:
            exp_fourier_fft = Coefficients.evaluate_Fourier_series(
                coefficients=fft_coeffs,
                frequencies=fft_freqs,
                inputs=ref_input,
            )

            exp_fourier = Coefficients.evaluate_Fourier_series(
                coefficients=analytical_coeffs[0],
                frequencies=analytical_freqs[0],
                inputs=ref_input,
            )

            exp_tree = coeff_tree(inputs=ref_input)

            assert np.isclose(
                exp_fourier_fft, exp_fourier, atol=1.0e-5
            ), "FFT and analytical Fourier series do not match"

            assert np.isclose(
                exp_tree, exp_fourier, atol=1.0e-5
            ), "Analytic Fourier series evaluation not working"


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
