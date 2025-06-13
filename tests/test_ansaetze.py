import random
from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit, Gates
import pennylane as qml
import pennylane.numpy as np
import pytest
import logging

logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_gate_bitflip_noise():
    dev = qml.device("default.mixed", wires=1, shots=3000)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliZ(0))

    no_noise = circuit({})
    with_noise = circuit({"BitFlip": 0.5})

    assert np.isclose(no_noise, -1, atol=0.1), (
        f"Expected ~-1 without noise, got {no_noise}"
    )
    assert np.isclose(with_noise, 0, atol=0.1), (
        f"Expected ~0 with noise, got {with_noise}"
    )


@pytest.mark.unittest
def test_gate_phaseflip_noise():
    dev = qml.device("default.mixed", wires=1, shots=1000)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.H(wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliX(0))

    no_noise = circuit({})
    with_noise = circuit({"PhaseFlip": 0.5})

    assert np.isclose(no_noise, 1, atol=0.1), (
        f"Expected ~1 with no noise, got {no_noise}"
    )
    assert np.isclose(with_noise, 0, atol=0.1), (
        f"Expected ~0 with PhaseFlip noise, got {with_noise}"
    )


@pytest.mark.unittest
def test_gate_depolarizing_noise():
    dev = qml.device("default.mixed", wires=1, shots=3000)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliZ(0))

    no_noise = circuit({})
    with_noise = circuit({"Depolarizing": 3 / 4})

    assert np.isclose(no_noise, -1, atol=0.1), (
        f"Expected ~-1 with no noise, got {no_noise}"
    )
    assert np.isclose(with_noise, 0, atol=0.1), (
        f"Expected ~0 with Depolarizing noise, got {with_noise}"
    )


@pytest.mark.unittest
def test_gate_nqubitdepolarizing_noise():
    dev_two = qml.device("default.mixed", wires=2, shots=3000)

    @qml.qnode(dev_two)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0)
        Gates.CRX(np.pi, wires=[0, 1], noise_params=noise_params)
        return qml.expval(qml.PauliZ(1))

    no_noise_two = circuit({})
    with_noise_two = circuit({"MultiQubitDepolarizing": 15 / 16})

    assert np.isclose(no_noise_two, -1, atol=0.1), (
        f"Expected ~-1 with no noise, got {no_noise_two}"
    )
    assert np.isclose(with_noise_two, 0, atol=0.1), (
        f"Expected ~0 with noise, got {with_noise_two}"
    )

    dev_three = qml.device("default.mixed", wires=3, shots=3000)

    @qml.qnode(dev_three)
    def circuit(noise_params=None):
        if noise_params is not None:
            Gates.NQubitDepolarizingChannel(
                noise_params.get("MultiQubitDepolarizing", 0), wires=[0, 1, 2]
            )

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    no_noise_three = circuit({})
    with_noise_three = circuit({"MultiQubitDepolarizing": 63/64})

    assert np.isclose(no_noise_three, 1, atol=0.1), (
        f"Expected ~1 with no noise, got {no_noise_three}"
    )
    assert np.isclose(with_noise_three, 0, atol=0.1), (
        f"Expected ~0 with noise, got {with_noise_three}"
    )
