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
    # Results taken from: https://doi.org/10.1002/qute.201900070
    circuits = [9, 1, 2, 16, 3, 18, 10, 12, 15, 17, 4, 11, 7, 8, 19, 5, 13, 14, 6]
    results_n_layers_1 = [
        0.6773,
        0.2999,
        0.2860,
        0.2602,
        0.2396,
        0.2340,
        0.2286,
        0.1984,
        0.1892,
        0.1359,
        0.1343,
        0.1312,
        0.0977,
        0.0858,
        0.0809,
        0.0602,
        0.0516,
        0.0144,
        0.0043,
    ]

    results_n_layers_3 = [
        0.0322,
        0.2079,
        0.0084,
        0.0375,
        0.0403,
        0.0221,
        0.1297,
        0.0089,
        0.1152,
        0.0180,
        0.0107,
        0.0038,
        0.0162,
        0.0122,
        0.0040,
        0.0030,
        0.0049,
        0.0035,
        0.0039,
    ]

    # Circuits [5,7,8,11,12,13,14] are not included in the test cases,
    # because not implemented in ansaetze.py

    # Circuit 10 excluded because implementation with current setup not possible
    skip_indices = [5, 7, 8, 11, 12, 13, 14, 10]
    # skip_indices = [16, 3, 18, 10, 12, 15, 17, 4, 11, 7, 8, 19, 5, 13, 14, 6]

    test_cases = []
    for i, result1, result3 in zip(circuits, results_n_layers_1, results_n_layers_3):
        if i in skip_indices:
            continue
        test_cases.append(
            {
                "circuit_type": f"Circuit_{i}",
                "n_qubits": 4,
                "n_layers": 1,
                "result": result1,
            }
        )
        # test_cases.append({
        #     "circuit_type": f"Circuit_{i}",
        #     "n_qubits": 4,
        #     "n_layers": 3,
        #     "result": result3,
        # })

    tolerance = 0.35  # FIXME: reduce when reason for discrepancy is found
    kl_distances: list[tuple[int, float]] = []
    for test_case in test_cases:
        print(f"--- Running Expressibility test for {test_case['circuit_type']} ---")
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            initialization_domain=[0, 2 * np.pi],
            data_reupload=False,
        )

        _, _, z = Expressibility.state_fidelities(
            seed=1000,
            n_bins=75,
            n_samples=5000,
            model=model,
            scale=False,
        )

        _, y_haar = Expressibility.haar_integral(
            n_qubits=test_case["n_qubits"],
            n_bins=75,
            cache=False,
            scale=False,
        )

        # Calculate the mean (over all inputs, if required)
        kl_dist = Expressibility.kullback_leibler_divergence(z, y_haar).mean()

        circuit_number = int(test_case["circuit_type"].split("_")[1])
        kl_distances.append((circuit_number, kl_dist.item()))

        difference = abs(kl_dist - test_case["result"])
        if math.isclose(difference, 0.0, abs_tol=1e-10):
            error = 0
        else:
            error = abs(kl_dist - test_case["result"]) / (test_case["result"])

        print(
            f"KL Divergence: {kl_dist}, Expected Result: {test_case['result']}, Error: {error}"
        )
        assert (
            error < tolerance
        ), f"Expressibility of circuit {test_case['circuit_type']} is not\
            {test_case['result']} but {kl_dist} instead.\
            Deviation {(error*100):.1f}>{tolerance*100}%"

    expected_results = sorted(
        [
            (circuit, results_n_layers_1[circuits.index(circuit)])
            for circuit in circuits
            if circuit not in skip_indices
        ],
        key=lambda x: x[1],
    )

    actual_results = sorted(kl_distances, key=lambda x: x[1])

    print("Expected \t| Actual")
    for expected, actual in zip(expected_results, actual_results):
        print(f"{expected[0]}, {expected[1]} \t| {actual[0]}, {actual[1]}")
    assert [circuit for circuit, _ in expected_results] == [
        circuit for circuit, _ in actual_results
    ]


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
        cache=False,
        scale=True,
    )

    assert y.shape == (8,)

    # _ = Expressibility.kullback_leibler_divergence(z, y)


if __name__ == "__main__":
    test_expressibility()
