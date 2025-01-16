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
    # circuits = [1, 7, 3, 16, 8, 5, 18, 17, 4, 10, 19, 13, 12, 14, 11, 6, 2, 15, 9]
    # ent_res = [0.0000, 0.3246, 0.3424, 0.3464, 0.3932, 0.4099, 0.4383, 0.4541,
    # 0.4715, 0.5369, 0.5937, 0.6070, 0.6487, 0.6613, 0.7330, 0.7803, 0.8083,
    # 0.8186, 1.0000]

    # Circuits [5,7,8,11,12,13,14] are not included in the test cases,
    # because not implemented in ansaetze.py
    test_cases = [
        {
            "circuit_type": "No_Entangling",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.0,
        },
        {
            "circuit_type": "Strongly_Entangling",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.4032,
        },
        {
            "circuit_type": "Circuit_1",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.0000,
        },
        {
            "circuit_type": "Circuit_3",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.3424,
        },
        {
            "circuit_type": "Circuit_16",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.3464,
        },
        {
            "circuit_type": "Circuit_18",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.4383,
        },
        {
            "circuit_type": "Circuit_17",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.4541,
        },
        {
            "circuit_type": "Circuit_4",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.4715,
        },
        {
            "circuit_type": "Circuit_10",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.5369,
        },
        {
            "circuit_type": "Circuit_19",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.5937,
        },
        {
            "circuit_type": "Circuit_6",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.7803,
        },
        {
            "circuit_type": "Circuit_2",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.8083,
        },
        {
            "circuit_type": "Circuit_15",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 0.8186,
        },
        {
            "circuit_type": "Circuit_9",
            "n_qubits": 2,
            "n_layers": 1,
            "result": 1.0000,
        },
    ]

    for test_case in test_cases:
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

        difference = abs(ent_cap - test_case["result"])
        if math.isclose(difference, 0.0, abs_tol=1e-3):
            error = 0
        else:
            error = abs(ent_cap - test_case["result"]) / (test_case["result"])

        assert (
            error < 0.1
        ), f"Entangling-capability of circuit {test_case['circuit_type']} is not {test_case['result']}\
            for circuit ansatz {test_case['circuit_type']}.\
            Was {ent_cap} instead. Deviation {error*100:1f}>10%"


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


if __name__ == "__main__":
    test_entanglement()
