from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Circuit, Gates
from qml_essentials.utils import QuanTikz
import pytest
import logging
import inspect
import shutil
import os
import hashlib
from typing import Optional
import pennylane as qml
import pennylane.numpy as np

from typing import List, Callable

logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_parameters() -> None:
    test_cases = [
        # {
        #     "shots": None,
        #     "execution_type": "expval",
        #     "output_qubit": 0,
        #     "force_mean": False,
        #     "exception": False,
        # },
        # {
        #     "shots": None,
        #     "execution_type": "expval",
        #     "output_qubit": -1,
        #     "force_mean": False,
        #     "exception": False,
        # },
        # {
        #     "shots": None,
        #     "execution_type": "expval",
        #     "output_qubit": -1,
        #     "force_mean": True,
        #     "exception": False,
        # },
        # {
        #     "shots": None,
        #     "execution_type": "density",
        #     "output_qubit": 0,
        #     "force_mean": False,
        #     "exception": False,
        # },
        # {
        #     "shots": 1024,
        #     "execution_type": "probs",
        #     "output_qubit": 0,
        #     "force_mean": False,
        #     "exception": False,
        # },
        {
            "shots": 1024,
            "execution_type": "probs",
            "output_qubit": -1,
            "force_mean": True,
            "exception": False,
        },
        {
            "shots": 1024,
            "execution_type": "expval",
            "output_qubit": 0,
            "force_mean": False,
            "exception": False,
        },
        {
            "shots": 1024,
            "execution_type": "expval",
            "output_qubit": 0,
            "force_mean": True,
            "exception": False,
        },
        {
            "shots": 1024,
            "execution_type": "density",
            "output_qubit": 0,
            "force_mean": False,
            "exception": True,
        },
    ]

    # Test the most minimal call
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )
    assert (model() == model(model.params)).all()

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            output_qubit=test_case["output_qubit"],
            shots=test_case["shots"],
        )

        if test_case["exception"]:
            with pytest.warns(UserWarning):
                _ = model(
                    model.params,
                    inputs=None,
                    execution_type=test_case["execution_type"],
                    force_mean=test_case["force_mean"],
                )
        else:
            result = model(
                model.params,
                inputs=None,
                execution_type=test_case["execution_type"],
                force_mean=test_case["force_mean"],
            )

            if test_case["shots"] is None:
                assert hasattr(
                    result, "requires_grad"
                ), "No 'requires_grad' property available in output."
            else:
                # TODO: not supported by PennyLane yet
                pass
            if test_case["output_qubit"] == -1:
                if test_case["force_mean"]:
                    assert (
                        result.size == 1 or result.shape[0] == 1
                    ), f"Shape of {test_case['output_qubit']} is not correct."
                else:
                    if test_case["execution_type"] == "expval":
                        # check for 2 because of n qubits
                        assert (
                            result.shape[0] == 2
                        ), f"Shape of {test_case['output_qubit']} is not correct."
                    elif test_case["execution_type"] == "probs":
                        assert (
                            result.shape[0] == 4
                        ), f"Shape of {test_case['output_qubit']} is not correct."
            str(model)


@pytest.mark.smoketest
def test_encoding() -> None:
    test_cases = [
        {"encoding_unitary": Gates.RX, "type": Callable, "input": [0]},
        {
            "encoding_unitary": [Gates.RX, Gates.RY],
            "type": List,
            "input": [[0, 0]],
        },
        {"encoding_unitary": "RX", "type": Callable, "input": [0]},
        {"encoding_unitary": ["RX", "RY"], "type": List, "input": [[0, 0]]},
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            encoding=test_case["encoding_unitary"],
        )
        _ = model(
            model.params,
            inputs=test_case["input"],
        )

        assert isinstance(model._enc, test_case["type"])


@pytest.mark.unittest
def test_cache() -> None:
    # Stupid try removing caches
    try:
        shutil.rmtree(".cache")
    except Exception as e:
        logger.warning(e)

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    result = model(
        model.params,
        inputs=None,
        cache=True,
    )

    hs = hashlib.md5(
        repr(
            {
                "n_qubits": model.n_qubits,
                "n_layers": model.n_layers,
                "pqc": model.pqc.__class__.__name__,
                "dru": model.data_reupload,
                "params": model.params,
                "noise_params": model.noise_params,
                "execution_type": model.execution_type,
                "inputs": np.array([[0]]),
                "output_qubit": model.output_qubit,
            }
        ).encode("utf-8")
    ).hexdigest()

    cache_folder: str = ".cache"
    if not os.path.exists(cache_folder):
        raise Exception("Cache folder does not exist.")

    name: str = f"pqc_{hs}.npy"
    file_path: str = os.path.join(cache_folder, name)

    if os.path.isfile(file_path):
        cached_result = np.load(file_path)

    assert np.array_equal(
        result, cached_result
    ), "Cached result and calcualted result is not equal."


@pytest.mark.expensive
@pytest.mark.smoketest
def test_lightning() -> None:
    model = Model(
        n_qubits=12,  # model.lightning_threshold
        n_layers=1,
        circuit_type="Hardware_Efficient",
    )
    assert model.circuit.device.name == "lightning.qubit"

    _ = model(
        model.params,
        inputs=None,
    )


@pytest.mark.smoketest
def test_draw() -> None:
    model = Model(
        n_qubits=4,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        output_qubit=-1,
        # initialization="pi",  # Initialize all parameters to pi
    )

    # Set specific parameter values for better visualization
    # model.params = np.array(
    #     [[np.pi / 2, np.pi / 4, np.pi / 3, np.pi / 6] * 3]
    # )  # 3 parameters per qubit in Hardware_Efficient

    repr(model)
    _ = model.draw(figure=True)
    quantikz_str = model.draw(tikz=True, gate_values=True)
    QuanTikz.export(quantikz_str=quantikz_str, destination="./tikz_test.tex")
    model.draw(figure=True)[0].savefig("circuit.png")


@pytest.mark.smoketest
def test_initialization() -> None:
    test_cases = [
        {
            "initialization": "random",
        },
        {
            "initialization": "zeros",
        },
        {
            "initialization": "zero-controlled",
        },
        {
            "initialization": "pi-controlled",
        },
        {
            "initialization": "pi",
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization=test_case["initialization"],
            output_qubit=0,
            shots=1024,
        )

        _ = model(
            model.params,
            inputs=None,
            noise_params=None,
            cache=False,
            execution_type="expval",
        )


@pytest.mark.unittest
def test_re_initialization() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        initialization_domain=[-2 * np.pi, 0],
        random_seed=1000,
    )

    assert model.params.max() <= 0, "Parameters should be in [-2pi, 0]!"

    temp_params = model.params.copy()

    model.initialize_params(rng=np.random.default_rng(seed=1001))

    assert not np.allclose(
        model.params, temp_params, atol=1e-3
    ), "Re-Initialization failed!"


@pytest.mark.smoketest
def test_ansaetze() -> None:
    ansatz_cases = Ansaetze.get_available()

    for ansatz in ansatz_cases:
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
                "ThermalRelaxation": {"T1": 2000.0, "T2": 1000.0, "t_factor": 1},
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
                    Gates.CZ(wires=[q, q + 1], noise_params=noise_params)

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
def test_multi_input() -> None:
    input_cases = [
        np.random.rand(1, 1),
        np.random.rand(1, 2),
        np.random.rand(1, 3),
        np.random.rand(2, 1),
        np.random.rand(3, 2),
        np.random.rand(20, 1),
    ]
    input_cases = [2 * np.pi * i for i in input_cases]
    input_cases.append(None)

    for inputs in input_cases:
        logger.info(
            f"Testing input with shape: "
            f"{inputs.shape if inputs is not None else 'None'}"
        )
        encoding = (
            Gates.RX if inputs is None else [Gates.RX for _ in range(inputs.shape[1])]
        )
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization="random",
            encoding=encoding,
            output_qubit=0,
            shots=1024,
        )

        out = model(
            model.params,
            inputs=inputs,
            noise_params=None,
            cache=False,
            execution_type="expval",
        )

        if inputs is not None:
            if len(out.shape) > 0:
                assert out.shape[0] == inputs.shape[0], (
                    f"batch dimension mismatch, expected {inputs.shape[0]} "
                    f"as an output dimension, but got {out.shape[0]}"
                )
            else:
                assert (
                    inputs.shape[0] == 1
                ), "expected one elemental input for zero dimensional output"
        else:
            assert len(out.shape) == 0, "expected one elemental output for empty input"


@pytest.mark.unittest
def test_dru() -> None:
    test_cases = [
        {
            "dru": False,
            "degree": 1,
        },
        {
            "dru": True,
            "degree": 4,
        },
        {
            "dru": [[True, False], [False, True]],
            "degree": 2,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=2,
            circuit_type="Circuit_19",
            data_reupload=test_case["dru"],
            initialization="random",
            output_qubit=0,
            shots=1024,
        )

        assert (
            model.degree == test_case["degree"]
        ), f"Expected degree {test_case['degree']} but got {model.degree}\
            for dru {test_case['dru']}"

        _ = model(
            model.params,
            inputs=None,
            noise_params=None,
            cache=False,
            execution_type="expval",
        )


@pytest.mark.unittest
def test_local_state() -> None:
    test_cases = [
        {
            "noise_params": None,
            "execution_type": "density",
        },
        {
            "noise_params": {
                "BitFlip": 0.1,
                "PhaseFlip": 0.2,
                "AmplitudeDamping": 0.3,
                "PhaseDamping": 0.4,
                "Depolarizing": 0.5,
            },
            "execution_type": "density",
        },
        {
            "noise_params": None,
            "execution_type": "expval",
        },
    ]

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        data_reupload=True,
        initialization="random",
        output_qubit=0,
    )

    # Check default values
    assert model.noise_params is None
    assert model.execution_type == "expval"

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization="random",
            output_qubit=0,
        )

        model.noise_params = test_case["noise_params"]
        model.execution_type = test_case["execution_type"]

        _ = model(
            model.params,
            inputs=None,
            noise_params=None,
            cache=False,
        )

        # check if setting "externally" is working
        assert model.noise_params == test_case["noise_params"]
        assert model.execution_type == test_case["execution_type"]

        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization="random",
            output_qubit=0,
        )

        _ = model(
            model.params,
            inputs=None,
            cache=False,
            noise_params=test_case["noise_params"],
            execution_type=test_case["execution_type"],
        )

        # check if setting in the forward call is working
        assert model.noise_params == test_case["noise_params"]
        assert model.execution_type == test_case["execution_type"]


@pytest.mark.unittest
def test_local_and_global_meas() -> None:
    test_cases = [
        {
            "inputs": None,
            "execution_type": "expval",
            "output_qubit": -1,
            "shots": None,
            "out_shape": (2,),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": -1,
            "shots": None,
            "out_shape": (2, 3),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": 0,
            "shots": None,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": [0, 1],
            "shots": None,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "inputs": None,
            "execution_type": "density",
            "output_qubit": -1,
            "shots": None,
            "out_shape": (4, 4),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "output_qubit": -1,
            "shots": None,
            "out_shape": (3, 4, 4),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "output_qubit": 0,
            "shots": None,
            "out_shape": (3, 4, 4),
            "warning": True,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "output_qubit": -1,
            "shots": 1024,
            "out_shape": (3, 4),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "output_qubit": 0,
            "shots": 1024,
            "out_shape": (3, 2),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "output_qubit": [0, 1],
            "shots": 1024,
            "out_shape": (3, 4),
            "warning": False,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization="random",
            output_qubit=test_case["output_qubit"],
            shots=test_case["shots"],
        )
        if test_case["warning"]:
            with pytest.warns(UserWarning):
                out = model(
                    model.params,
                    inputs=test_case["inputs"],
                    noise_params=None,
                    cache=False,
                    execution_type=test_case["execution_type"],
                )
        else:
            out = model(
                model.params,
                inputs=test_case["inputs"],
                noise_params=None,
                cache=False,
                execution_type=test_case["execution_type"],
            )

        assert (
            out.shape == test_case["out_shape"]
        ), f"Expected {test_case['out_shape']}, got shape {out.shape}\
            for test case {test_case}"


@pytest.mark.unittest
def test_parity() -> None:
    model_a = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
        output_qubit=[0, 1],  # parity
    )
    model_b = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
        output_qubit=-1,  # individual
    )

    result_a = model_a(params=model_a.params, inputs=None, force_mean=True)
    result_b = model_b(
        params=model_a.params, inputs=None, force_mean=True
    )  # use same params!

    assert not np.allclose(
        result_a, result_b
    ), f"Models should be different! Got {result_a} and {result_b}"


@pytest.mark.smoketest
def test_params_store() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
    )
    opt = qml.AdamOptimizer(stepsize=0.01)

    def cost(params):
        return model(params=params, inputs=np.array([0])).mean()._value

    params, cost = opt.step_and_cost(cost, model.params)


@pytest.mark.unittest
def test_pauli_circuit_model() -> None:
    test_cases = [
        {
            "shots": None,
            "output_qubit": 0,
            "force_mean": False,
            "inputs": np.array([0.1, 0.2, 0.3]),
        },
        {
            "shots": None,
            "output_qubit": -1,
            "force_mean": False,
            "inputs": np.array([0.1, 0.2, 0.3]),
        },
        {
            "shots": None,
            "output_qubit": -1,
            "force_mean": True,
            "inputs": np.array([0.1, 0.2, 0.3]),
        },
        {
            "shots": 1024,
            "output_qubit": 0,
            "force_mean": False,
            "inputs": np.array([0.1, 0.2, 0.3]),
        },
        {
            "shots": 1024,
            "output_qubit": 0,
            "force_mean": True,
            "inputs": np.array([0.1, 0.2, 0.3]),
        },
        {
            "shots": None,
            "output_qubit": 0,
            "force_mean": False,
            "inputs": None,
        },
        {
            "shots": None,
            "output_qubit": -1,
            "force_mean": False,
            "inputs": None,
        },
        {
            "shots": None,
            "output_qubit": -1,
            "force_mean": True,
            "inputs": None,
        },
        {
            "shots": 1024,
            "output_qubit": 0,
            "force_mean": False,
            "inputs": None,
        },
        {
            "shots": 1024,
            "output_qubit": 0,
            "force_mean": True,
            "inputs": None,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=3,
            n_layers=2,
            circuit_type="Circuit_19",
            output_qubit=test_case["output_qubit"],
            shots=test_case["shots"],
            as_pauli_circuit=True,
        )

        pauli_model = Model(
            n_qubits=3,
            n_layers=2,
            circuit_type="Circuit_19",
            output_qubit=test_case["output_qubit"],
            shots=test_case["shots"],
            as_pauli_circuit=True,
        )

        result_circuit = model(
            model.params,
            inputs=test_case["inputs"],
            force_mean=test_case["force_mean"],
        )

        result_pauli_circuit = pauli_model(
            pauli_model.params,
            inputs=test_case["inputs"],
            force_mean=test_case["force_mean"],
        )

        assert all(
            np.isclose(result_circuit, result_pauli_circuit, atol=1e-5).flatten()
        ), (
            f"results of Pauli Circuit and basic Ansatz should be equal, but "
            f"are {result_pauli_circuit} and {result_circuit} for testcase "
            f"{test_case}, respectively."
        )
