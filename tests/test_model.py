from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze
import pytest
import numpy as np
import logging
import inspect
import shutil
import os
import hashlib


logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_parameters() -> None:
    test_cases = [
        {
            "shots": -1,
            "execution_type": "expval",
            "output_qubit": 0,
            "force_mean": False,
            "exception": False,
        },
        {
            "shots": -1,
            "execution_type": "expval",
            "output_qubit": -1,
            "force_mean": False,
            "exception": False,
        },
        {
            "shots": -1,
            "execution_type": "expval",
            "output_qubit": -1,
            "force_mean": True,
            "exception": False,
        },
        {
            "shots": -1,
            "execution_type": "density",
            "output_qubit": 0,
            "force_mean": False,
            "exception": False,
        },
        {
            "shots": 1024,
            "execution_type": "probs",
            "output_qubit": 0,
            "force_mean": False,
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
            "execution_type": "density",
            "output_qubit": 0,
            "force_mean": False,
            "exception": True,
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

        if test_case["exception"]:
            with pytest.warns(UserWarning):
                _ = model(
                    model.params,
                    inputs=None,
                    noise_params=None,
                    cache=False,
                    execution_type=test_case["execution_type"],
                    force_mean=test_case["force_mean"],
                )
        else:
            result = model.__call__(
                model.params,
                inputs=None,
                noise_params=None,
                cache=False,
                execution_type=test_case["execution_type"],
                force_mean=test_case["force_mean"],
            )

            if test_case["shots"] < 0:
                assert hasattr(
                    result, "requires_grad"
                ), "No 'requires_grad' property available in output."

            if test_case["output_qubit"] == -1:
                if test_case["force_mean"]:
                    assert (
                        result.shape[0] == 1
                    ), f"Shape of {test_case['output_qubit']} is not correct."
                else:
                    # check for 2 because of n qubits
                    assert (
                        result.shape[0] == 2
                    ), f"Shape of {test_case['output_qubit']} is not correct."
            str(model)


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
                "inputs": None,
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


@pytest.mark.smoketest
def test_ansaetze() -> None:
    ansatz_cases = Ansaetze.get_available()

    for ansatz in ansatz_cases:
        # Skipping Circuit_15, as it is not yet correctly implemented
        if ansatz.__name__ == "Circuit_15":
            continue

        logger.info(f"Testing Ansatz: {ansatz.__name__}")
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type=ansatz.__name__,
            data_reupload=True,
            initialization="random",
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
        np.random.rand(1),
        np.random.rand(1, 1),
        np.random.rand(1, 2),
        np.random.rand(1, 3),
        np.random.rand(2, 1),
        np.random.rand(20, 1),
    ]
    input_cases = [2 * np.pi * i for i in input_cases]
    input_cases.append(None)

    for inputs in input_cases:
        logger.info(
            f"Testing input with shape: "
            f"{inputs.shape if inputs is not None else 'None'}"
        )
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization="random",
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


@pytest.mark.smoketest
def test_dru() -> None:
    dru_cases = [False, True]

    for dru in dru_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=dru,
            initialization="random",
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
                "DepolarizingChannel": 0.5,
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
            "shots": -1,
            "out_shape": (2, 1),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": -1,
            "shots": -1,
            "out_shape": (2, 3),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": 0,
            "shots": -1,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": [0, 1],
            "shots": -1,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "inputs": None,
            "execution_type": "density",
            "output_qubit": -1,
            "shots": -1,
            "out_shape": (4, 4),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "output_qubit": -1,
            "shots": -1,
            "out_shape": (3, 4, 4),
            "warning": False,
        },
        {
            "inputs": np.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "output_qubit": 0,
            "shots": -1,
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
