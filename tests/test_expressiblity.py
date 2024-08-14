from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients
from qml_essentials.entanglement import Entanglement
from qml_essentials.expessibility import Expressibility

import pytest
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


def test_divergence() -> None:
    test_cases = [
        {
            "n_qubits": 2,
            "n_bins": 10,
            "result": 0.000,
        },
    ]

    for test_case in test_cases:
        _, y_haar_a = Expressibility.haar_integral(
            n_qubits=test_case["n_qubits"],
            n_bins=test_case["n_bins"],
            cache=True,
        )

        # We also test here the chache functionality
        _, y_haar_b = Expressibility.haar_integral(
            n_qubits=test_case["n_qubits"],
            n_bins=test_case["n_bins"],
            cache=False,
        )

        # Calculate the mean (over all inputs, if required)
        kl_dist = Expressibility.kullback_leibler_divergence(y_haar_a, y_haar_b).mean()

        assert math.isclose(
            kl_dist.mean(), test_case["result"], abs_tol=1e-3
        ), f"Distance between two identical haar measures not equal."


def test_expressibility() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 2,
            "n_layers": 3,
            "n_bins": 10,
            "n_samples": 200,
            "n_input_samples": 2,
            "result": 1.858,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 2,
            "n_layers": 3,
            "n_bins": 10,
            "n_samples": 200,
            "n_input_samples": 2,
            "result": 2.629,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=True,
            initialization="random",
            output_qubit=0,
        )

        _, _, z = Expressibility.state_fidelities(
            n_bins=test_case["n_bins"],
            n_samples=test_case["n_samples"],
            n_input_samples=test_case["n_input_samples"],
            seed=1000,
            model=model,
            cache=False,
        )

        _, y_haar = Expressibility.haar_integral(
            n_qubits=test_case["n_qubits"],
            n_bins=test_case["n_bins"],
            cache=False,
        )

        # Calculate the mean (over all inputs, if required)
        kl_dist = Expressibility.kullback_leibler_divergence(z, y_haar).mean()

        assert math.isclose(
            kl_dist.mean(), test_case["result"], abs_tol=1e-3
        ), f"Expressibility is not {test_case['result']}\
            for circuit ansatz {test_case['circuit_type']}.\
            Was {kl_dist} instead"
