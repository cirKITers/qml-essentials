from qml_essentials.model import Model
import pytest


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

            str(model)


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
