import random
from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit, Gates, PulseGates
import pennylane as qml
import pennylane.numpy as np
import pytest
import logging
import jax
jax.config.update("jax_enable_x64", True)


logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_gate_gateerror_noise():
    dev = qml.device("default.mixed", wires=1, shots=3000)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliZ(0))

    no_noise = circuit({})
    with_noise = circuit({"GateError": 100})

    assert np.isclose(no_noise, -1, atol=0.1), (
        f"Expected ~-1 with no noise, got {no_noise}"
    )
    assert not np.isclose(with_noise, no_noise, atol=0.1), (
        "Expected with noise output to differ,"
        + f"got with noise: {with_noise} and with no noise: {no_noise}"
    )


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
        f"Expected ~-1 with no noise, got {no_noise}"
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


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi/4, np.pi/2, np.pi])
def test_pulse_RX_gate(w):
    pg = PulseGates()
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def ideal_circuit():
        qml.RX(w, wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        pg.RX(w, wires=0)
        return qml.state()

    state_ideal = ideal_circuit()
    state_pulse = pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low for w={w}: {fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
def test_pulse_RY_gate(w):
    pg = PulseGates()
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def ideal_circuit():
        qml.RY(w, wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        pg.RY(w, wires=0)
        return qml.state()

    state_ideal = ideal_circuit()
    state_pulse = pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low for w={w}: {fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
def test_pulse_RZ_gate(w):
    pg = PulseGates()
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def ideal_circuit():
        qml.Hadamard(wires=0)  # Prepare |+> so RZ acts non-trivially
        qml.RZ(w, wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.Hadamard(wires=0)
        pg.RZ(w, wires=0)
        return qml.state()

    state_ideal = ideal_circuit()
    state_pulse = pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low for w={w}: {fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


@pytest.mark.unittest
def test_pulse_H_gate():
    pg = PulseGates()
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def ideal_circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        pg.H(wires=0)
        return qml.state()

    state_ideal = ideal_circuit()
    state_pulse = pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low for H gate: {fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for H gate: {phase_diff}"


@pytest.mark.unittest
def test_pulse_CZ_gate():
    pg = PulseGates()
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def ideal_circuit():
        qml.H(wires=0)
        qml.H(wires=1)
        qml.CZ(wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        qml.H(wires=1)
        pg.CZ(wires=[0, 1])
        return qml.state()

    state_ideal = ideal_circuit()
    state_pulse = pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-1), f"Phase off: {phase_diff}"


@pytest.mark.unittest
def test_pulse_CNOT_gate():
    pg = PulseGates()
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def ideal_circuit():
        qml.H(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        pg.CNOT(wires=[0, 1])
        return qml.state()

    state_ideal = ideal_circuit()
    state_pulse = pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off: {phase_diff}"
