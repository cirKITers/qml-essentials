from qml_essentials.model import Model
from qml_essentials.expressibility import Expressibility

import pennylane.numpy as np
import logging
import math
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.unittest
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
        ), "Distance between two identical haar measures not equal."


@pytest.mark.unittest
@pytest.mark.expensive
def test_expressibility() -> None:
    test_cases = [
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 3,
            "n_layers": 1,
            "n_bins": 10,
            "n_samples": 400,
            "n_input_samples": 10,
            "result": 2.905,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 3,
            "n_layers": 1,
            "n_bins": 10,
            "n_samples": 400,
            "n_input_samples": 10,
            "result": 6.670,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
        )

        _, _, z = Expressibility.state_fidelities(
            seed=1000,
            n_bins=test_case["n_bins"],
            n_samples=test_case["n_samples"],
            n_input_samples=test_case["n_input_samples"],
            input_domain=[0, 2 * np.pi],
            model=model,
        )

        _, y_haar = Expressibility.haar_integral(
            n_qubits=test_case["n_qubits"],
            n_bins=test_case["n_bins"],
        )

        # Calculate the mean (over all inputs, if required)
        kl_dist = Expressibility.kullback_leibler_divergence(z, y_haar).mean()

        assert math.isclose(
            kl_dist.mean(), test_case["result"], abs_tol=1e-3
        ), f"Expressibility is not {test_case['result']}\
            for circuit ansatz {test_case['circuit_type']}.\
            Was {kl_dist} instead"


@pytest.mark.unittest
@pytest.mark.expensive
def test_scaling() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
    )

    _, _, z = Expressibility.state_fidelities(
        seed=1000,
        n_bins=4,
        n_samples=10,
        n_input_samples=0,
        input_domain=[0, 2 * np.pi],
        model=model,
        scale=True,
    )

    assert z.shape == (8,)

    _, y = Expressibility.haar_integral(
        n_qubits=model.n_qubits,
        n_bins=4,
        cache=True,
        scale=True,
    )

    assert y.shape == (8,)
