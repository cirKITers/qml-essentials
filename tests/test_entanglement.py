from qml_essentials.model import Model
from qml_essentials.entanglement import Entanglement

import logging
import math
import pytest

logger = logging.getLogger(__name__)


@pytest.mark.expensive
@pytest.mark.unittest
def test_entanglement() -> None:
    test_cases = [
        {
            "circuit_type": "No_Entangling",
            "n_qubits": 2,
            "n_layers": 1,
            "n_samples": 1000,
            "result": 0.0,
        },
        {
            "circuit_type": "Strongly_Entangling",
            "n_qubits": 2,
            "n_layers": 1,
            "n_samples": 2000,
            "result": 0.3912,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=test_case["n_qubits"],
            n_layers=test_case["n_layers"],
            circuit_type=test_case["circuit_type"],
            data_reupload=True,
            initialization="random",
        )

        ent_cap = Entanglement.meyer_wallach(
            model, n_samples=test_case["n_samples"], seed=1000, cache=False
        )

        assert math.isclose(
            ent_cap, test_case["result"], abs_tol=1e-3
        ), f"Entangling capacity is not {test_case['result']}\
            for circuit ansatz {test_case['circuit_type']}.\
            Was {ent_cap} instead"


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
