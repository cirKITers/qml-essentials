from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients

import numpy as np
import logging

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    test_coefficients()
