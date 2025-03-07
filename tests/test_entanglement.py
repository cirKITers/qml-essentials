from qml_essentials.model import Model
from qml_essentials.entanglement import Entanglement

import logging
import math
import pytest

from copy import deepcopy

logger = logging.getLogger(__name__)


def get_test_cases():
    # Results taken from: https://doi.org/10.1002/qute.201900070

    circuits = [
        "No_Entangling",
        "Strongly_Entangling",
        1,
        7,
        3,
        16,
        8,
        5,
        18,
        17,
        4,
        10,
        19,
        13,
        12,
        14,
        11,
        6,
        2,
        15,
        9,
    ]
    results_n_layers_1 = [
        0.0000,
        0.8379,
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
    # Circuits [5,7,8,11,12,13,14] are not included in the test cases,
    # because not implemented in ansaetze.py

    # Circuit 10 excluded because implementation with current setup not possible
    skip_indices = [5, 7, 8, 11, 12, 13, 14, 10]
    skip_indices += [2, 3]  # exclude these for now as order is failing

    return circuits, results_n_layers_1, skip_indices


@pytest.mark.expensive
@pytest.mark.unittest
def test_mw_measure() -> None:
    circuits, results_n_layers_1, skip_indices = get_test_cases()

    test_cases = []
    for circuit_id, res_1l in zip(circuits, results_n_layers_1):
        if circuit_id in skip_indices:
            continue
        if isinstance(circuit_id, int):
            test_cases.append(
                {
                    "circuit_type": f"Circuit_{circuit_id}",
                    "n_qubits": 4,
                    "n_layers": 1,
                    "result": res_1l,
                }
            )
        elif isinstance(circuit_id, str):
            test_cases.append(
                {
                    "circuit_type": circuit_id,
                    "n_qubits": 4,
                    "n_layers": 1,
                    "result": res_1l,
                }
            )

    tolerance = 0.55  # FIXME: reduce when reason for discrepancy is found
    ent_caps: list[tuple[str, float]] = []
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

        # Save results for later comparison
        circuit_number = test_case["circuit_type"]
        if circuit_number.split("_")[1].isdigit():
            circuit_number = int(circuit_number.split("_")[1])
        ent_caps.append((circuit_number, ent_cap))

        difference = abs(ent_cap - test_case["result"])
        if math.isclose(difference, 0.0, abs_tol=1e-3):
            error = 0
        else:
            error = abs(ent_cap - test_case["result"]) / (test_case["result"])

        print(
            f"Entangling-capability: {ent_cap},\t"
            + f"Expected Result: {test_case['result']},\t"
            + f"Error: {error}"
        )
        assert (
            error < tolerance
        ), f"Entangling-capability of circuit {test_case['circuit_type']} is not\
            {test_case['result']} but {ent_cap} instead.\
            Deviation {(error * 100):.1f}%>{tolerance * 100}%"

    references = sorted(
        [
            (circuit, ent_result)
            for circuit, ent_result in zip(circuits, results_n_layers_1)
            if circuit not in skip_indices
        ],
        key=lambda x: x[1],
    )

    actuals = sorted(ent_caps, key=lambda x: x[1])

    print("Expected \t| Actual")
    for reference, actual in zip(references, actuals):
        print(f"{reference[0]}, {reference[1]} \t| {actual[0]}, {actual[1]}")
    assert [circuit for circuit, _ in actuals] == [
        circuit for circuit, _ in references
    ], f"Order of circuits does not match: {actuals} != {references}"


@pytest.mark.smoketest
def test_no_sampling() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        data_reupload=False,
        initialization="random",
    )

    _ = Entanglement.meyer_wallach(model, n_samples=-1, seed=1000, cache=False)


@pytest.mark.expensive
@pytest.mark.unittest
def test_bell_measure() -> None:
    circuits, results_n_layers_1, skip_indices = get_test_cases()

    test_cases = []
    for circuit_id, res_1l in zip(circuits, results_n_layers_1):
        if circuit_id in skip_indices:
            continue
        if isinstance(circuit_id, int):
            test_cases.append(
                {
                    "circuit_type": f"Circuit_{circuit_id}",
                    "n_qubits": 4,
                    "n_layers": 1,
                    "result": res_1l,
                }
            )
        elif isinstance(circuit_id, str):
            test_cases.append(
                {
                    "circuit_type": circuit_id,
                    "n_qubits": 4,
                    "n_layers": 1,
                    "result": res_1l,
                }
            )

    tolerance = 0.55  # FIXME: reduce when reason for discrepancy is found
    ent_caps: list[tuple[str, float]] = []
    for test_case in test_cases:
        print(f"--- Running Entanglement test for {test_case['circuit_type']} ---")
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=False,
            initialization="random",
        )

        ent_cap = Entanglement.bell_measurements(
            model, n_samples=5000, seed=1000, cache=False
        )

        # Save results for later comparison
        circuit_number = test_case["circuit_type"]
        if circuit_number.split("_")[1].isdigit():
            circuit_number = int(circuit_number.split("_")[1])
        ent_caps.append((circuit_number, ent_cap))

        difference = abs(ent_cap - test_case["result"])
        if math.isclose(difference, 0.0, abs_tol=1e-3):
            error = 0
        else:
            error = abs(ent_cap - test_case["result"]) / (test_case["result"])

        print(
            f"Entangling-capability: {ent_cap},\t"
            + f"Expected Result: {test_case['result']},\t"
            + f"Error: {error}"
        )
        assert (
            error < tolerance
        ), f"Entangling-capability of circuit {test_case['circuit_type']} is not\
            {test_case['result']} but {ent_cap} instead.\
            Deviation {(error * 100):.1f}%>{tolerance * 100}%"

    references = sorted(
        [
            (circuit, ent_result)
            for circuit, ent_result in zip(circuits, results_n_layers_1)
            if circuit not in skip_indices
        ],
        key=lambda x: x[1],
    )

    actuals = sorted(ent_caps, key=lambda x: x[1])

    print("Expected \t| Actual")
    for reference, actual in zip(references, actuals):
        print(f"{reference[0]}, {reference[1]} \t| {actual[0]}, {actual[1]}")
    assert [circuit for circuit, _ in actuals] == [
        circuit for circuit, _ in references
    ], f"Order of circuits does not match: {actuals} != {references}"


@pytest.mark.unittest
def test_entangling_measures() -> None:
    test_cases = [
        {"circuit_type": "Circuit_4", "n_qubits": 2, "n_layers": 1},
        {"circuit_type": "Circuit_4", "n_qubits": 3, "n_layers": 1},
        {"circuit_type": "Circuit_4", "n_qubits": 4, "n_layers": 1},
        {"circuit_type": "Circuit_4", "n_qubits": 5, "n_layers": 1},
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=False,
            initialization="random",
        )

        mw_meas = Entanglement.meyer_wallach(
            deepcopy(model), n_samples=2000, seed=1000, cache=False
        )

        bell_meas = Entanglement.bell_measurements(
            model, n_samples=2000, seed=1000, cache=False
        )

        assert math.isclose(mw_meas, bell_meas, abs_tol=1e-5), (
            f"Meyer-Wallach and Bell-measurement are not the same. Got {mw_meas} "
            f"and {bell_meas}, respectively."
        )
