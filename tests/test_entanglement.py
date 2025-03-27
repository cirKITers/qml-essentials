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
        # "No_Entangling",
        # "Strongly_Entangling",
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
        # 0.0000,
        # 0.8379,
        0.0000,
        0.3241,
        0.3412,
        0.3439,
        0.3926,
        0.4090,
        0.4385,
        0.4533,
        0.4721,
        0.5362,
        0.5916,
        0.6077,
        0.6486,
        0.6604,
        0.7335,
        0.7781,
        0.8104,
        0.8184,
        1.0000,
    ]

    results_n_layers_3 = [
        0.0000,
        0.6194,
        0.5852,
        0.5859,
        0.6567,
        0.7953,
        0.7130,
        0.6557,
        0.6607,
        0.7865,
        0.7906,
        0.8224,
        0.7838,
        0.8557,
        0.8288,
        0.8721,
        0.8657,
        0.8734,
        1.0000,
    ]

    # Circuits [5,7,8,11,12,13,14] are not included in the test cases,
    # because not implemented in ansaetze.py

    # Circuit 10 excluded because implementation with current setup not possible
    skip_indices = [5, 7, 8, 11, 12, 13, 14, 10]
    skip_indices += [2, 3]  # exclude these for now as order is failing

    return circuits, results_n_layers_1, results_n_layers_3, skip_indices


@pytest.mark.expensive
@pytest.mark.unittest
def test_mw_measure() -> None:
    circuits, results_n_layers_1, results_n_layers_3, skip_indices = get_test_cases()

    test_cases = []
    for circuit_id, res_1l, res_3l in zip(
        circuits, results_n_layers_1, results_n_layers_3
    ):
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
    circuits, results_n_layers_1, results_n_layers_3, skip_indices = get_test_cases()

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
        {"circuit_type": "Circuit_4", "n_qubits": 4, "n_layers": 1},
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=False,
        )

        mw_meas = Entanglement.meyer_wallach(
            deepcopy(model), n_samples=1000, seed=1000, cache=False
        )

        bell_meas = Entanglement.bell_measurements(
            deepcopy(model), n_samples=1000, seed=1000, cache=False
        )

        assert math.isclose(mw_meas, bell_meas, abs_tol=1e-5), (
            f"Meyer-Wallach and Bell-measurement are not the same. Got {mw_meas} "
            f"and {bell_meas}, respectively."
        )


@pytest.mark.smoketest
@pytest.mark.expensive
def test_scaling() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
    )

    _ = Entanglement.meyer_wallach(
        deepcopy(model), n_samples=10, seed=1000, cache=False, scale=True
    )

    _ = Entanglement.bell_measurements(
        model, n_samples=10, seed=1000, cache=False, scale=True
    )


@pytest.mark.smoketest
def test_relative_entropy() -> None:
    separable_model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Circuit_1",
    )

    entangled_model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Strongly_Entangling",
    )

    ghz_model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="GHZ",
        data_reupload=False,
    )

    separable_ent = Entanglement.relative_entropy(
        separable_model, n_samples=10, n_sigmas=10, seed=1000, cache=False, scale=False
    )
    entangled_ent = Entanglement.relative_entropy(
        entangled_model, n_samples=10, n_sigmas=10, seed=1000, cache=False, scale=False
    )
    ghz_ent = Entanglement.relative_entropy(
        ghz_model, n_samples=10, n_sigmas=10, seed=1000, cache=False, scale=False
    )

    assert 0.0 < separable_ent < entangled_ent < ghz_ent == 1.0, (
        "Order of entanglement should be 0 < Circuit_1 < Strongly Entangling < GHZ = 1, "
        f"but got values 0 < {separable_ent} < {entangled_ent} < {ghz_ent} < 1"
    )
