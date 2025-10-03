from typing import Optional
from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit, Gates, UnitaryGates
from qml_essentials.ansaetze import PulseInformation as pinfo
import pennylane as qml
import pennylane.numpy as np
import jax
from jax import numpy as jnp
import pytest
import inspect
import logging

jax.config.update("jax_enable_x64", True)


logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_gate_gateerror_noise():
    UnitaryGates.rng = np.random.default_rng(1000)

    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)
        return qml.expval(qml.PauliZ(0))

    no_noise = circuit({})
    with_noise = circuit({"GateError": 50})

    assert np.isclose(
        no_noise, -1, atol=0.01
    ), f"Expected ~-1 with no noise, got {no_noise}"
    assert not np.isclose(with_noise, no_noise, atol=0.01), (
        "Expected with noise output to differ, "
        + f"got with noise: {with_noise} and with no noise: {no_noise}"
    )


@pytest.mark.unittest
def test_batch_gate_error():
    UnitaryGates.rng = np.random.default_rng(1000)

    model = Model(
        n_qubits=1,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    inputs = np.array([0.1, 0.1, 0.1, 0.1])
    res_a = model(inputs=inputs, noise_params={"GateError": 50})
    # check if each output is different
    assert not np.allclose(res_a, np.flip(res_a))

    UnitaryGates.batch_gate_error = False
    res_b = model(inputs=inputs, noise_params={"GateError": 50})
    # check if each output is the same
    assert np.allclose(res_b, np.flip(res_b)), (
        "Expected all outputs to be the same " "when batch_gate_error is False"
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
        def n_pulse_params_per_layer(n_qubits: int) -> int:
            n_params = pinfo.num_params("RY")
            n_params += pinfo.num_params("RZ")
            n_params *= n_qubits

            n_params += (n_qubits - 1) * pinfo.num_params("CRY")
            n_params += (n_qubits - 1) * pinfo.num_params("CY")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
            return None

        @staticmethod
        def build(w: np.ndarray, n_qubits: int, **kwargs):
            w_idx = 0
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            for q in range(n_qubits - 1):
                Gates.CRY(w[w_idx], wires=[q, q + 1], **kwargs)
                Gates.CY(wires=[q + 1, q], **kwargs)
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


@pytest.mark.unittest
@pytest.mark.parametrize(
    "w",
    [
        (0.0, 0.0, 0.0),  # Identity
        (np.pi / 2, 0.0, 0.0),  # Pure RX
        (0.0, np.pi / 2, 0.0),  # Pure RY
        (np.pi, np.pi / 2, np.pi),  # Mixed rotation
    ],
)
def test_pulse_Rot_gate(w):
    phi, theta, omega = w

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.Rot(phi, theta, omega, wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        Gates.Rot(phi, theta, omega, wires=0, gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("Rot")
        Gates.Rot(
            phi, theta, omega, wires=0, pulse_params=pulse_params, gate_mode="pulse"
        )
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(
        fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for w={w}: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse w={w}: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
def test_pulse_RX_gate(w):
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.RX(w, wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        Gates.RX(w, wires=0, gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("RX")
        Gates.RX(w, wires=0, pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(
        fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for w={w}: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse w={w}: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
def test_pulse_RY_gate(w):
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.RY(w, wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        Gates.RY(w, wires=0, gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("RY")
        Gates.RY(w, wires=0, pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(
        fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for w={w}: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse w={w}: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
def test_pulse_RZ_gate(w):
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.Hadamard(wires=0)  # Prepare |+> so RZ acts non-trivially
        qml.RZ(w, wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.Hadamard(wires=0)
        Gates.RZ(w, wires=0, gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("RZ")
        qml.Hadamard(wires=0)
        Gates.RZ(w, wires=0, pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(
        fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for w={w}: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse w={w}: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


@pytest.mark.unittest
def test_pulse_H_gate():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.Hadamard(wires=0)
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        Gates.H(wires=0, gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("H")
        Gates.H(wires=0, pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(
        fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for H gate: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse H gate: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for H gate: {phase_diff}"


@pytest.mark.unittest
def test_pulse_CZ_gate():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.H(wires=0)
        qml.H(wires=1)
        qml.CZ(wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        qml.H(wires=1)
        Gates.CZ(wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CZ")
        qml.H(wires=0)
        qml.H(wires=1)
        Gates.CZ(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-1), f"Phase off: {phase_diff}"


@pytest.mark.unittest
def test_pulse_CY_gate():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.H(wires=0)
        qml.CY(wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        Gates.CY(wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CY")
        qml.H(wires=0)
        Gates.CY(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off: {phase_diff}"


@pytest.mark.unittest
def test_pulse_CX_gate():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.H(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        Gates.CX(wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CX")
        qml.H(wires=0)
        Gates.CX(wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off: {phase_diff}"


# TODO: Unskip CRZ, CRY, CRX tests when their optimization is fixed
@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
# @pytest.mark.skip(reason="CRZ not properly optimized, low fidelity")
def test_pulse_CRZ_gate(w):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.H(wires=0)
        qml.H(wires=1)
        qml.CRZ(w, wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        qml.H(wires=1)
        Gates.CRZ(w, wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CRZ")
        qml.H(wires=0)
        qml.H(wires=1)
        Gates.CRZ(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-1), f"Phase off: {phase_diff}"


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
# @pytest.mark.skip(reason="CRY not properly optimized, low fidelity")
def test_pulse_CRY_gate(w):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.H(wires=0)
        qml.CRY(w, wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        Gates.CRY(w, wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CRY")
        qml.H(wires=0)
        Gates.CRY(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off: {phase_diff}"


@pytest.mark.unittest
@pytest.mark.parametrize("w", [np.pi / 4, np.pi / 2, np.pi])
# @pytest.mark.skip(reason="CRX not properly optimized, low fidelity")
def test_pulse_CRX_gate(w):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def unitary_circuit():
        qml.H(wires=0)
        qml.CRX(w, wires=[0, 1])
        return qml.state()

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        Gates.CRX(w, wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CRX")
        qml.H(wires=0)
        Gates.CRX(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_ideal = unitary_circuit()
    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    fidelity = np.abs(np.vdot(state_ideal, state_pulse)) ** 2
    assert np.isclose(fidelity, 1.0, atol=1e-2), f"Fidelity too low: {fidelity}"

    custom_fidelity = np.abs(np.vdot(state_ideal, state_custom_pulse)) ** 2
    assert np.isclose(
        custom_fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for custom pulse: {custom_fidelity}"

    phase_diff = np.angle(np.vdot(state_ideal, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off: {phase_diff}"


# TODO: Remove CRZ, CRY, CRX smoketests when their optimization is fixed
@pytest.mark.smoketest
@pytest.mark.parametrize("w", [np.pi])
def test_pulse_CRZ_gate_smoke(w):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        qml.H(wires=1)
        Gates.CRZ(w, wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CRZ")
        qml.H(wires=0)
        qml.H(wires=1)
        Gates.CRZ(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    assert state_pulse is not None
    assert state_custom_pulse is not None


@pytest.mark.smoketest
@pytest.mark.parametrize("w", [np.pi])
def test_pulse_CRY_gate_smoke(w):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        Gates.CRY(w, wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CRY")
        qml.H(wires=0)
        Gates.CRY(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    assert state_pulse is not None
    assert state_custom_pulse is not None


@pytest.mark.smoketest
@pytest.mark.parametrize("w", [np.pi])
def test_pulse_CRX_gate_smoke(w):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def pulse_circuit():
        qml.H(wires=0)
        Gates.CRX(w, wires=[0, 1], gate_mode="pulse")
        return qml.state()

    @qml.qnode(dev)
    def custom_pulse_circuit():
        pulse_params = pinfo.optimized_params("CRX")
        qml.H(wires=0)
        Gates.CRX(w, wires=[0, 1], pulse_params=pulse_params, gate_mode="pulse")
        return qml.state()

    state_pulse = pulse_circuit()
    state_custom_pulse = custom_pulse_circuit()

    assert state_pulse is not None
    assert state_custom_pulse is not None


@pytest.mark.unittest
def test_invalid_pulse_params():
    invalid_type_pulse_params = [
        np.array(["10", 5, "1"]),
        [10, 5, "1"],
        (10, 5, "1"),
    ]

    for pp in invalid_type_pulse_params:
        with pytest.raises(TypeError):
            Gates.RX(np.pi, 0, pulse_params=pp, gate_mode="pulse")

    invalid_len_pulse_params = [jnp.array([10, 5, 1, 1]), [10, 10, 5, 5, 1, 1], (10,)]

    for pp in invalid_len_pulse_params:
        with pytest.raises(ValueError):
            Gates.RX(np.pi, 0, pulse_params=pp, gate_mode="pulse")
