from typing import Optional
from qml_essentials.qoc import QOC
from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit
from qml_essentials.gates import Gates, UnitaryGates
from qml_essentials.gates import PulseInformation as pinfo
from qml_essentials import yaqsi as ys
from qml_essentials import operations as op
import numpy as np
import jax
from jax import numpy as jnp
import pytest
import inspect
import itertools
import time

import logging

jax.config.update("jax_enable_x64", True)


logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_gate_gateerror_noise():
    random_key = jax.random.key(1000)

    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params, random_key=random_key)

    obs = [op.PauliZ(wires=0)]

    script = ys.Script(circuit, n_qubits=1)
    no_noise = script.execute(type="expval", obs=obs, args=({},))
    with_noise = script.execute(type="expval", obs=obs, args=({"GateError": 50},))

    assert np.isclose(
        no_noise, -1, atol=0.01
    ), f"Expected ~-1 with no noise, got {no_noise}"
    assert not np.isclose(with_noise, no_noise, atol=0.01), (
        "Expected with noise output to differ, "
        + f"got with noise: {with_noise} and with no noise: {no_noise}"
    )


@pytest.mark.unittest
def test_batch_gate_error():
    model = Model(
        n_qubits=1,
        n_layers=1,
        circuit_type="Circuit_1",
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
        circuit_type="Circuit_1",
    )
    # should raise error if gate error is not filtered out correctly
    # as density operations would then run on sv simulator
    model(noise_params={"GateError": 0.5})


@pytest.mark.unittest
def test_gate_bitflip_noise():
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)

    obs = [op.PauliZ(wires=0)]

    script = ys.Script(circuit, n_qubits=1)
    no_noise = script.execute(type="expval", obs=obs, args=({},))
    with_noise = script.execute(type="expval", obs=obs, args=({"BitFlip": 0.5},))

    assert np.isclose(
        no_noise, -1, atol=0.1
    ), f"Expected ~-1 with no noise, got {no_noise}"
    assert np.isclose(
        with_noise, 0, atol=0.1
    ), f"Expected ~0 with noise, got {with_noise}"


@pytest.mark.unittest
def test_gate_phaseflip_noise():
    def circuit(noise_params=None):
        Gates.H(wires=0, noise_params=noise_params)

    obs = [op.PauliX(wires=0)]

    script = ys.Script(circuit, n_qubits=1)
    no_noise = script.execute(type="expval", obs=obs, args=({},))
    with_noise = script.execute(type="expval", obs=obs, args=({"PhaseFlip": 0.5},))

    assert np.isclose(
        no_noise, 1, atol=0.1
    ), f"Expected ~1 with no noise, got {no_noise}"
    assert np.isclose(
        with_noise, 0, atol=0.1
    ), f"Expected ~0 with PhaseFlip noise, got {with_noise}"


@pytest.mark.unittest
def test_gate_depolarizing_noise():
    def circuit(noise_params=None):
        Gates.RX(np.pi, wires=0, noise_params=noise_params)

    obs = [op.PauliZ(wires=0)]

    script = ys.Script(circuit, n_qubits=1)
    no_noise = script.execute(type="expval", obs=obs, args=({},))
    with_noise = script.execute(type="expval", obs=obs, args=({"Depolarizing": 3 / 4},))

    assert np.isclose(
        no_noise, -1, atol=0.1
    ), f"Expected ~-1 with no noise, got {no_noise}"
    assert np.isclose(
        with_noise, 0, atol=0.1
    ), f"Expected ~0 with Depolarizing noise, got {with_noise}"


@pytest.mark.unittest
def test_gate_nqubitdepolarizing_noise():
    def circuit_two(noise_params=None):
        Gates.RX(np.pi, wires=0)
        Gates.CRX(np.pi, wires=[0, 1], noise_params=noise_params)

    obs_two = [op.PauliZ(wires=1)]

    script_two = ys.Script(circuit_two, n_qubits=2)
    no_noise_two = script_two.execute(type="expval", obs=obs_two, args=({},))
    with_noise_two = script_two.execute(
        type="expval", obs=obs_two, args=({"MultiQubitDepolarizing": 15 / 16},)
    )

    assert np.isclose(
        no_noise_two, -1, atol=0.1
    ), f"Expected ~-1 with no noise, got {no_noise_two}"
    assert np.isclose(
        with_noise_two, 0, atol=0.1
    ), f"Expected ~0 with noise, got {with_noise_two}"

    def circuit_three(noise_params=None):
        if noise_params is not None:
            Gates.NQubitDepolarizingChannel(
                noise_params.get("MultiQubitDepolarizing", 0), wires=[0, 1, 2]
            )

    obs_three = [ys.build_parity_observable([0, 1, 2])]

    script_three = ys.Script(circuit_three, n_qubits=3)
    no_noise_three = script_three.execute(type="expval", obs=obs_three, args=({},))
    with_noise_three = script_three.execute(
        type="expval",
        obs=obs_three,
        args=({"MultiQubitDepolarizing": 63 / 64},),
    )

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
    # FIXME: un-ignore the following circuits
    ignore = [
        "No_Ansatz",
        "Circuit_5",
        "Circuit_6",
        "Circuit_7",
        "Circuit_8",
        "Circuit_13",
        "Circuit_14",
    ]

    for ansatz in Ansaetze.get_available():
        ansatz = ansatz.__name__

        if ansatz in ignore:
            continue
        model = Model(n_qubits=4, n_layers=1, circuit_type=ansatz, data_reupload=False)

        # slice the first (only) layer of this model to get the params per layer
        ctrl_params = model.pqc.get_control_angles(model.params[0], model.n_qubits)

        if ansatz in control_params.keys():
            # the ctrl params must be equal to the last two params in the set,
            # i.e. the params that go into the crx gates of Circuit 19
            assert np.allclose(
                ctrl_params, model.params[0, control_params[ansatz] :]
            ), f"Ctrl. params are not returned as expected for circuit {ansatz}."
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
        execution_type="density",
    )

    with pytest.warns(UserWarning):
        _ = model(
            model.params,
            inputs=None,
            noise_params={
                "UnsupportedNoise": 0.1,
            },
            execution_type="density",
        )


@pytest.mark.unittest
def test_min_qubit_warning() -> None:
    with pytest.warns(UserWarning):
        _ = Model(
            n_qubits=1,
            n_layers=1,
            circuit_type="Circuit_19",
        )


@pytest.mark.expensive
@pytest.mark.unittest
def test_pulse_params_ansaetze() -> None:
    test_cases = {
        "No_Ansatz": [1.0, 1.0],
        "Circuit_1": [-0.06347358, -0.99916184],
        "Circuit_2": [0.06294326, -0.99916164],
        "Circuit_3": [0.42394627, -0.41875092],
        "Circuit_4": [-0.15574438, -0.41875081],
        "Circuit_5": [-0.8643505, 0.22041861],
        "Circuit_6": [-0.48522045, -0.38787328],
        "Circuit_7": [-0.44584206, 0.4983159],
        "Circuit_8": [-0.44595841, 0.48103752],
        "Circuit_9": [0.00012548, -0.00023631],
        "Circuit_10": [0.91743331, -0.4636175],
        "Circuit_13": [0.81335946, -0.92070053],
        "Circuit_14": [0.60749882, -0.54199932],
        "Circuit_15": [-0.06616174, -0.06790603],
        "Circuit_16": [0.42394627, -0.41875092],
        "Circuit_17": [-0.15574438, -0.41875081],
        "Circuit_18": [0.08187592, -0.99445816],
        "Circuit_19": [-0.50660179, -0.95456148],
        "Circuit_20": [0.99820156, -0.06618526],
        "No_Entangling": [-0.99444818, 0.64505527],
        "Strongly_Entangling": [0.02930848, 0.60783151],
        "Hardware_Efficient": [-0.9407261, 0.47868613],
        "GHZ": [0.00026971, 0.00026978],
    }
    for ansatz, res in test_cases.items():
        logger.info(f"Testing Ansatz: {ansatz}")
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type=ansatz,
            data_reupload=False,
        )

        try:
            res = model(gate_mode="pulse")
            assert np.allclose(res, res, atol=1e-6)
        except Exception as e:
            raise Exception(f"Error for ansatz {ansatz}: {e}")


def test_pulse_benchmarks() -> None:
    start = time.time()
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        data_reupload=False,
    )
    res = model(gate_mode="pulse")
    end = time.time()
    print(f"Time: {end - start}")


@pytest.mark.unittest
def test_available_ansaetze() -> None:
    ansatze = set(Ansaetze.get_available())

    actual_ansaetze = set(
        ansatz for ansatz in Ansaetze.__dict__.values() if inspect.isclass(ansatz)
    )
    # check that the classes are the ones returned by .__subclasses__
    assert actual_ansaetze == ansatze


single_qubit_pulse_testdata = itertools.product(
    ["RX", "RY", "RZ", "H"], [np.pi / 4, np.pi / 2, np.pi]
)


@pytest.mark.unittest
@pytest.mark.parametrize("gate,w", single_qubit_pulse_testdata)
def test_single_qubit_pulse_gate(gate, w):
    qoc = QOC()
    pulse_circuit, target_circuit = getattr(qoc, "create_" + gate)()
    pulse_script = ys.Script(pulse_circuit, n_qubits=1)
    target_script = ys.Script(target_circuit, n_qubits=1)

    state_pulse = pulse_script.execute(
        type="state", args=(w, pinfo.gate_by_name(gate).params)
    )
    state_target = target_script.execute(type="state", args=(w,))

    fidelity = jnp.abs(jnp.vdot(state_target, state_pulse)) ** 2
    assert np.isclose(
        fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for w={w}: {fidelity}"

    phase_diff = np.angle(np.vdot(state_target, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


two_qubit_pulse_testdata = itertools.product(
    ["CX", "CY", "CZ", "CRX", "CRY", "CRZ"], [np.pi / 4, np.pi / 2, np.pi]
)


@pytest.mark.unittest
@pytest.mark.parametrize("gate,w", two_qubit_pulse_testdata)
def test_two_qubit_pulse_gate(gate, w):
    qoc = QOC()
    pulse_circuit, target_circuit = getattr(qoc, "create_" + gate)()
    pulse_script = ys.Script(pulse_circuit, n_qubits=2)
    target_script = ys.Script(target_circuit, n_qubits=2)

    state_pulse = pulse_script.execute(
        type="state", args=(w, pinfo.gate_by_name(gate).params)
    )
    state_target = target_script.execute(type="state", args=(w,))

    fidelity = jnp.abs(jnp.vdot(state_target, state_pulse)) ** 2
    assert np.isclose(
        fidelity, 1.0, atol=1e-2
    ), f"Fidelity too low for w={w}: {fidelity}"

    phase_diff = np.angle(np.vdot(state_target, state_pulse))
    assert np.isclose(phase_diff, 0.0, atol=1e-2), f"Phase off for w={w}: {phase_diff}"


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
