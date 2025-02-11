from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients
from pennylane.fourier import coefficients as pcoefficients

import numpy as np
import logging
import pytest

from functools import partial


logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_coefficients() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 1,
            "n_layers": 3,
            "n_bins": 10,
            "n_samples": 200,
            "n_input_samples": 2,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "n_bins": 10,
            "n_samples": 200,
            "n_input_samples": 2,
            "output_qubit": 0,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 4,
            "n_layers": 1,
            "n_bins": 10,
            "n_samples": 200,
            "n_input_samples": 2,
            "output_qubit": [0, 1, 2, 3],
        },
    ]

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

        assert len(coeffs) == model.degree * 2 + 1, "Wrong number of coefficients"
        assert np.isclose(
            np.sum(coeffs).imag, 0.0, rtol=1.0e-5
        ), "Imaginary part is not zero"

        partial_circuit = partial(model, model.params)
        ref_coeffs = pcoefficients(partial_circuit, 1, model.degree)

        assert np.allclose(
            coeffs, ref_coeffs, rtol=1.0e-5
        ), "Coefficients don't match the pennylane reference"


@pytest.mark.unittest
def test_oversampling_time() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    assert (
        Coefficients.sample_coefficients(model, mts=2).shape[0] == 10
    ), "Oversampling time failed"


@pytest.mark.unittest
def test_oversampling_frequency() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    assert (
        Coefficients.sample_coefficients(model, mfs=4).shape[0] == 9
    ), "Oversampling frequency failed"


@pytest.mark.unittest
def test_shift() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )
    coeffs = Coefficients.sample_coefficients(model, shift=True)

    assert (
        np.abs(coeffs) == np.abs(coeffs[::-1])
    ).all(), "Shift failed. Spectrum must be symmetric."
