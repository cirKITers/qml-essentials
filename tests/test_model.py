from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze
import pytest
import numpy as np
import logging
import inspect

logger = logging.getLogger(__name__)


def test_parameters() -> None:
    test_cases = [
        {
            "shots": -1,
            "execution_type": "expval",
            "exception": False,
        },
        {
            "shots": -1,
            "execution_type": "density",
            "exception": False,
        },
        {
            "shots": 1024,
            "execution_type": "probs",
            "exception": False,
        },
        {
            "shots": 1024,
            "execution_type": "expval",
            "exception": False,
        },
        {
            "shots": 1024,
            "execution_type": "density",
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
            output_qubit=0,
            shots=test_case["shots"],
        )

        if test_case["exception"]:
            with pytest.warns(UserWarning):
                result = model(
                    model.params,
                    inputs=None,
                    noise_params=None,
                    cache=False,
                    execution_type=test_case["execution_type"],
                )
        else:
            _ = model(
                model.params,
                inputs=None,
                noise_params=None,
                cache=False,
                execution_type=test_case["execution_type"],
            )


def test_cache() -> None:
    test_cases = [
        {
            "shots": 1024,
            "execution_type": "expval",
            "shape": (),
        },
        {
            "shots": -1,
            "execution_type": "density",
            "shape": (4, 4),
        },
        {
            "shots": 1024,
            "execution_type": "probs",
            "shape": (2,),
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization="random",
            output_qubit=0,
            shots=test_case["shots"],
        )

        result = model(
            model.params,
            inputs=None,
            noise_params=None,
            cache=True,
            execution_type=test_case["execution_type"],
        )

        assert result.shape == test_case["shape"], f"Test case: {test_case} failed"


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


def test_ansaetze() -> None:
    ansatz_cases = Ansaetze.get_available()

    for ansatz in ansatz_cases:
        # Skipping Circuit_15, as it is not yet correctly implemented
        if ansatz.__name__ == "Circuit_15":
            continue

        logger.info(f"Testing Ansatz: {ansatz.__name__}")
        model = Model(
            n_qubits=4,
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

        logger.info(f"{str(model)}")


def test_available_ansaetze() -> None:
    ansatze = set(Ansaetze.get_available())

    actual_ansaetze = set(
        ansatz for ansatz in Ansaetze.__dict__.values() if inspect.isclass(ansatz)
    )
    # check that the classes are the ones returned by .__subclasses__
    assert actual_ansaetze == ansatze


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
                ), f"expected one elemental input for zero dimensional output"
        else:
            assert len(out.shape) == 0, f"expected one elemental output for empty input"


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


def test_local_and_global_meas() -> None:
    inputs = np.array([0.1, 0.2, 0.3])
    test_cases = [
        {
            "execution_type": "expval",
            "output_qubit": -1,
            "shots": -1,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "execution_type": "expval",
            "output_qubit": 0,
            "shots": -1,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "execution_type": "expval",
            "output_qubit": [0, 1],
            "shots": -1,
            "out_shape": (2, 3),
            "warning": False,
        },
        {
            "execution_type": "density",
            "output_qubit": -1,
            "shots": -1,
            "out_shape": (3, 4, 4),
            "warning": False,
        },
        {
            "execution_type": "density",
            "output_qubit": 0,
            "shots": -1,
            "out_shape": (3, 4, 4),
            "warning": True,
        },
        {
            "execution_type": "probs",
            "output_qubit": -1,
            "shots": 1024,
            "out_shape": (3, 4),
            "warning": False,
        },
        {
            "execution_type": "probs",
            "output_qubit": 0,
            "shots": 1024,
            "out_shape": (3, 2),
            "warning": False,
        },
        {
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
                    inputs=inputs,
                    noise_params=None,
                    cache=False,
                    execution_type=test_case["execution_type"],
                )
        else:
            out = model(
                model.params,
                inputs=inputs,
                noise_params=None,
                cache=False,
                execution_type=test_case["execution_type"],
            )

        assert (
            out.shape == test_case["out_shape"]
        ), f"{test_case['execution_type']}: {out}"
