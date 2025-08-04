from typing import Optional
from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit, Gates
import pennylane as qml
import pennylane.numpy as np
import pytest
import inspect

import logging

logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_gate_gateerror_noise():
    Gates.rng = np.random.default_rng(1000)

    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliZ(0))

    no_noise = circuit({})
    with_noise = circuit({"GateError": 0.5})

    assert np.isclose(
        no_noise, -1, atol=0.01
    ), f"Expected ~-1 with no noise, got {no_noise}"
    assert not np.isclose(with_noise, no_noise, atol=0.01), (
        "Expected with noise output to differ, "
        + f"got with noise: {with_noise} and with no noise: {no_noise}"
    )


@pytest.mark.smoketest
def test_coherent_as_expval():
    model = Model(
        n_qubits=1,
        n_layers=1,
        circuit_type="Circuit_19",
    )
    # should raise error if gate error is not filtered out correctly
    # as density operations would then run on sv simulator
    model(noise_params={"GateError": 0.5})


@pytest.mark.unittest
def test_gate_bitflip_noise():
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliZ(0))

    no_noise = circuit({})
    with_noise = circuit({"BitFlip": 0.5})

    assert np.isclose(
        no_noise, -1, atol=0.1
    ), f"Expected ~-1 with no noise, got {no_noise}"
    assert np.isclose(
        with_noise, 0, atol=0.1
    ), f"Expected ~0 with noise, got {with_noise}"


@pytest.mark.unittest
def test_gate_phaseflip_noise():
    dev = qml.device("default.mixed", wires=1, shots=1000)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.H(wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliX(0))

    no_noise = circuit({})
    with_noise = circuit({"PhaseFlip": 0.5})

    assert np.isclose(
        no_noise, 1, atol=0.1
    ), f"Expected ~1 with no noise, got {no_noise}"
    assert np.isclose(
        with_noise, 0, atol=0.1
    ), f"Expected ~0 with PhaseFlip noise, got {with_noise}"


@pytest.mark.unittest
def test_gate_depolarizing_noise():
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliZ(0))

    no_noise = circuit({})
    with_noise = circuit({"Depolarizing": 3 / 4})

    assert np.isclose(
        no_noise, -1, atol=0.1
    ), f"Expected ~-1 with no noise, got {no_noise}"
    assert np.isclose(
        with_noise, 0, atol=0.1
    ), f"Expected ~0 with Depolarizing noise, got {with_noise}"


@pytest.mark.unittest
def test_gate_nqubitdepolarizing_noise():
    dev_two = qml.device("default.mixed", wires=2)

    @qml.qnode(dev_two)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0)
        Gates.CRX(np.pi, wires=[0, 1], noise_params=noise_params)
        return qml.expval(qml.PauliZ(1))

    no_noise_two = circuit({})
    with_noise_two = circuit({"MultiQubitDepolarizing": 15 / 16})

    assert np.isclose(
        no_noise_two, -1, atol=0.1
    ), f"Expected ~-1 with no noise, got {no_noise_two}"
    assert np.isclose(
        with_noise_two, 0, atol=0.1
    ), f"Expected ~0 with noise, got {with_noise_two}"

    dev_three = qml.device("default.mixed", wires=3)

    @qml.qnode(dev_three)
    def circuit(noise_params=None):
        if noise_params is not None:
            Gates.NQubitDepolarizingChannel(
                noise_params.get("MultiQubitDepolarizing", 0), wires=[0, 1, 2]
            )

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    no_noise_three = circuit({})
    with_noise_three = circuit({"MultiQubitDepolarizing": 63 / 64})

    assert np.isclose(
        no_noise_three, 1, atol=0.1
    ), f"Expected ~1 with no noise, got {no_noise_three}"
    assert np.isclose(
        with_noise_three, 0, atol=0.1
    ), f"Expected ~0 with noise, got {with_noise_three}"


@pytest.mark.unittest
def test_control_angles():
    control_params = {
        "Circuit_3": -3,
        "Circuit_4": -3,
        "Circuit_16": -3,
        "Circuit_17": -3,
        "Circuit_18": -4,
        "Circuit_19": -4,
    }
    ignore = ["No_Ansatz", "Circuit_6"]

    for ansatz in Ansaetze.get_available():
        ansatz = ansatz.__name__
        model = Model(n_qubits=4, n_layers=1, circuit_type=ansatz, data_reupload=False)

        # slice the first (only) layer of this model to get the params per layer
        ctrl_params = model.pqc.get_control_angles(model.params[0], model.n_qubits)

        if ansatz in control_params.keys():
            # the ctrl params must be equal to the last two params in the set,
            # i.e. the params that go into the crx gates of Circuit 19
            assert np.allclose(
                ctrl_params, model.params[0, control_params[ansatz] :]
            ), f"Ctrl. params are not returned as expected for circuit {ansatz}."
        elif ansatz in ignore:
            continue
        else:
            assert (
                ctrl_params.size == 0
            ), f"No ctrl. params expected for circuit {ansatz}"


@pytest.mark.smoketest
def test_ansaetze() -> None:
    for ansatz in Ansaetze.get_available():
        logger.info(f"Testing Ansatz: {ansatz.__name__}")
        model = Model(
            n_qubits=4,
            n_layers=1,
            circuit_type=ansatz.__name__,
            data_reupload=False,
            initialization="random",
            output_qubit=0,
            shots=1024,
        )

        _ = model(
            model.params,
            inputs=None,
            noise_params={
                "GateError": 0.1,
                "BitFlip": 0.1,
                "PhaseFlip": 0.2,
                "AmplitudeDamping": 0.3,
                "PhaseDamping": 0.4,
                "Depolarizing": 0.5,
                "MultiQubitDepolarizing": 0.6,
                "ThermalRelaxation": {"t1": 2000.0, "t2": 1000.0, "t_factor": 1},
                "StatePreparation": 0.1,
                "Measurement": 0.1,
            },
            cache=False,
            execution_type="density",
        )

    class custom_ansatz(Circuit):
        @staticmethod
        def n_params_per_layer(n_qubits: int) -> int:
            return n_qubits * 3

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, noise_params=None):
            w_idx = 0
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
                w_idx += 1

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    Gates.CRY(w[w_idx], wires=[q, q + 1], noise_params=noise_params)
                    Gates.CY(wires=[q + 1, q], noise_params=noise_params)
                    w_idx += 1

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type=custom_ansatz,
        data_reupload=True,
        initialization="random",
        output_qubit=0,
        shots=1024,
    )
    logger.info(f"{str(model)}")

    _ = model(
        model.params,
        inputs=None,
        noise_params={
            "GateError": 0.1,
            "PhaseFlip": 0.2,
            "AmplitudeDamping": 0.3,
            "Depolarizing": 0.5,
            "MultiQubitDepolarizing": 0.6,
        },
        cache=False,
        execution_type="density",
    )

    with pytest.warns(UserWarning):
        _ = model(
            model.params,
            inputs=None,
            noise_params={
                "UnsupportedNoise": 0.1,
            },
            cache=False,
            execution_type="density",
        )


@pytest.mark.unittest
def test_available_ansaetze() -> None:
    ansatze = set(Ansaetze.get_available())

    actual_ansaetze = set(
        ansatz for ansatz in Ansaetze.__dict__.values() if inspect.isclass(ansatz)
    )
    # check that the classes are the ones returned by .__subclasses__
    assert actual_ansaetze == ansatze
