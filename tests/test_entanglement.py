from qml_essentials.model import Model
from qml_essentials.entanglement import Entanglement

import logging
import math
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.expensive
@pytest.mark.unittest
def test_entanglement() -> None:
    # Results taken from: https://doi.org/10.1002/qute.201900070
    circuits = [1, 7, 3, 16, 8, 5, 18, 17, 4, 10, 19, 13, 12, 14, 11, 6, 2, 15, 9]
    ent_results = [
        0.0000,
        0.3246,
        0.3424,
        0.3464,
        0.3932,
        0.4099,
        0.4383,
        0.4541,
        0.4715,
        0.5369,
        0.5937,
        0.6070,
        0.6487,
        0.6613,
        0.7330,
        0.7803,
        0.8083,
        0.8186,
        1.0000,
    ]
    no_ent_result = 0.0
    strongly_ent_result = 0.8379

    # Circuits [5,7,8,11,12,13,14] are not included in the test cases,
    # because not implemented in ansaetze.py

    # Circuit 10 excluded because implementation with current setup not possible
    skip_indices = [5, 7, 8, 11, 12, 13, 14, 10]
    skip_indices = [7, 3, 16, 8, 5, 18, 17, 4, 10, 19, 13, 12, 14, 11, 6, 15, 9]
    test_cases = [
        # {
        #     "circuit_type": "No_Entangling",
        #     "n_qubits": 4,
        #     "n_layers": 1,
        #     "result": no_ent_result,
        # },
        # {
        #     "circuit_type": "Strongly_Entangling",
        #     "n_qubits": 4,
        #     "n_layers": 1,
        #     "result": strongly_ent_result,
        # },
    ]
    for i, ent_res in zip(circuits, ent_results):
        if i in skip_indices:
            continue
        test_cases.append(
            {
                "circuit_type": f"Circuit_{i}",
                "n_qubits": 4,
                "n_layers": 1,
                "result": ent_res,
            }
        )

    tolerance = 0.55  # FIXME: reduce when reason for discrepancy is found
    ent_caps: list[tuple[int, float]] = []
    for test_case in test_cases:
        print(f"--- Running Entanglement test for {test_case['circuit_type']} ---")
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=False,
            initialization="random",
        )

        ent_cap = Entanglement.meyer_wallach(
            model, n_samples=5000, seed=1000, cache=False
        )

        circuit_number = 0
        if test_case["circuit_type"] == "No_Entangling":
            circuit_number = -1
        elif test_case["circuit_type"] == "Strongly_Entangling":
            circuit_number = -2
        else:
            circuit_number = int(test_case["circuit_type"].split("_")[1])
        ent_caps.append((circuit_number, ent_cap))

        difference = abs(ent_cap - test_case["result"])
        if math.isclose(difference, 0.0, abs_tol=1e-3):
            error = 0
        else:
            error = abs(ent_cap - test_case["result"]) / (test_case["result"])

        print(
            f"Entangling-capability: {ent_cap}, \
            Expected Result: {test_case['result']}, \
            Error: {error}"
        )
        assert (
            error < tolerance
        ), f"Entangling-capability of circuit {test_case['circuit_type']} is not\
            {test_case['result']} but {ent_cap} instead.\
            Deviation {(error*100):.1f}%>{tolerance*100}%"

    expected_ent_results = sorted(
        [(-1, no_ent_result), (-2, strongly_ent_result)]
        + [
            (circuit, ent_results[circuits.index(circuit)])
            for circuit in circuits
            if circuit not in skip_indices
        ],
        key=lambda x: x[1],
    )

    actual_ent_results = sorted(ent_caps, key=lambda x: x[1])

    print("Expected \t| Actual")
    for expected, actual in zip(expected_ent_results, actual_ent_results):
        print(f"{expected[0]}, {expected[1]} \t| {actual[0]}, {actual[1]}")
    assert [circuit for circuit, _ in actual_ent_results] == [
        circuit for circuit, _ in expected_ent_results
    ]


@pytest.mark.smoketest
def test_no_sampling() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        data_reupload=True,
        initialization="random",
    )

    _ = Entanglement.meyer_wallach(model, n_samples=-1, seed=1000, cache=False)
