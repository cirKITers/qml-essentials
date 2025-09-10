import random
from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Gates
import pytest
import logging
import shutil
import os
import hashlib
import pennylane as qml
import pennylane.numpy as np
import time

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
            "shots": None,
            "execution_type": "state",
            "output_qubit": -1,
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
                    elif test_case["execution_type"] == "state":
                        assert (
                            result.shape[0] == 4
                        ), f"Shape of {test_case['output_qubit']} is not correct."
            str(model)


@pytest.mark.unittest
def test_trainable_frequencies() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        trainable_frequencies=True,
    )

    assert model.enc_params.requires_grad, "Encoding parameters enc_params require grad"

    # setting test data
    domain = [-np.pi, np.pi]
    omegas = np.array([1.2, 2.6, 3.4, 4.9])
    coefficients = np.array([0.5, 0.5, 0.5, 0.5])
    n_d = int(np.ceil(2 * np.max(np.abs(domain)) * np.max(omegas)))
    x = np.linspace(domain[0], domain[1], num=n_d)

    def f(x):
        return 1 / np.linalg.norm(omegas) * np.sum(coefficients * np.cos(omegas.T * x))

    y = np.stack([f(sample) for sample in x])

    def cost_fct(params, enc_params):
        y_hat = model(params=params, enc_params=enc_params, inputs=x, force_mean=True)
        return np.mean((y_hat - y) ** 2)

    opt = qml.AdamOptimizer(stepsize=0.01)
    enc_params_before = model.enc_params.copy()
    (model.params, model.enc_params), cost_val = opt.step_and_cost(
        cost_fct, model.params, model.enc_params
    )
    enc_params_after = model.enc_params.copy()

    assert not np.allclose(
        enc_params_before, enc_params_after
    ), "enc_params did not update during training"

    grads = qml.grad(cost_fct, argnum=1)(model.params, model.enc_params)
    assert np.any(np.abs(grads) > 1e-6), "Gradient wrt enc_params is too small"

    # Smoketest to check model outside training
    model(enc_params=np.array(model.enc_params, requires_grad=False))
    model.trainable_frequencies = False
    model(enc_params=np.array(model.enc_params, requires_grad=False))


@pytest.mark.unittest
def test_transform_input() -> None:
    domain = [-1, 1]
    omegas = np.array([1, 2, 3, 4])
    n_d = int(np.ceil(2 * np.max(np.abs(domain)) * np.max(omegas)))
    x = np.linspace(domain[0], domain[1], num=n_d)

    model = Model(
        n_qubits=1,
        n_layers=1,
        circuit_type="No_Ansatz",
        encoding="RX",
        data_reupload=False,
    )

    # Test the intended use of transform_input()
    inputs = np.array([[0.5, -0.2]])
    enc_params = np.array([2.0, 3.0])

    # Test for qubit 0, feature 0
    result = model.transform_input(inputs, enc_params)
    expected = enc_params * inputs
    assert np.allclose(result, expected), "Incorrect transform for qubit 0"

    # Test modified transform_input()
    model.transform_input = lambda inputs, enc_params: (np.arccos(inputs))

    result_new = model(model.params, x)

    assert np.allclose(x, result_new), "model.transform_input does not work as intended"


@pytest.mark.unittest
def test_batching() -> None:
    for ansatz in Ansaetze.get_available():
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type=ansatz.__name__,
        )

        n_samples = 3
        model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)
        params = model.params

        res = np.zeros((n_samples, 4, 4), dtype=np.complex128)
        for i in range(n_samples):
            res[i] = model(params=params[:, :, i], execution_type="density")

        assert res.shape == (n_samples, 4, 4), "Shape of batching is not correct"
        assert (
            res == model(params=params, execution_type="density")
        ).all(), "Content of batching is not equal"

    ## Multi-Dim Input

    # model = Model(
    #     n_qubits=2,
    #     n_layers=1,
    #     circuit_type="Circuit_15",
    #     encoding=[Gates.RX, Gates.RX],
    # )

    # n_samples = 3
    # model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)
    # params = model.params

    # res = np.zeros((n_samples, 4, 4), dtype=np.complex128)
    # for i in range(n_samples):
    #     res[i] = model(
    #         inputs=np.random.rand(1, 2),
    #         params=params[:, :, i],
    #         execution_type="density",
    #     )

    # assert res.shape == (n_samples, 4, 4), "Shape of batching is not correct"
    # assert (
    #     res == model(params=params, execution_type="density")
    # ).all(), "Content of batching is not equal"


@pytest.mark.unittest
def test_multiprocessing_density() -> None:
    # use n_samples that is not a multiple of the threshold
    n_samples = 1500

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        mp_threshold=500,
    )

    model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_parallel = model(params=params, execution_type="density")
    t_parallel = time.time() - start

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_single = model(params=params, execution_type="density")
    t_single = time.time() - start

    assert (
        t_parallel < t_single
    ), "Time required for multiprocessing larger than single process"

    assert (
        res_parallel.shape == res_single.shape
    ), "Shape of multiprocessing is not correct"
    assert (res_parallel == res_single).all(), "Content of multiprocessing is not equal"


@pytest.mark.unittest
def test_multiprocessing_expval() -> None:
    n_samples = 40000  # expval requires more samples for advantage

    model = Model(
        n_qubits=6,  # .. and larger circuits
        n_layers=6,
        circuit_type="Circuit_19",
        mp_threshold=4000,
    )

    model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_parallel = model(params=params, execution_type="expval")
    t_parallel = time.time() - start

    model = Model(
        n_qubits=6,
        n_layers=6,
        circuit_type="Circuit_19",
    )

    model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_single = model(params=params, execution_type="expval")
    t_single = time.time() - start

    assert (
        t_parallel < t_single
    ), "Time required for multiprocessing larger than single process"

    print(t_parallel, t_single)
    assert (
        res_parallel.shape == res_single.shape
    ), "Shape of multiprocessing is not correct"
    assert (res_parallel == res_single).all(), "Content of multiprocessing is not equal"


@pytest.mark.smoketest
def test_state_preparation() -> None:
    test_cases = [
        {
            "state_preparation_unitary": Gates.H,
        },
        {
            "state_preparation_unitary": [Gates.H, Gates.H],
        },
        {
            "state_preparation_unitary": "H",
        },
        {
            "state_preparation_unitary": ["H", "H"],
        },
        {
            "state_preparation_unitary": None,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            state_preparation=test_case["state_preparation_unitary"],
            remove_zero_encoding=False,
        )

        _ = model(
            model.params,
        )


@pytest.mark.smoketest
def test_encoding() -> None:
    test_cases = [
        {
            "encoding_unitary": Gates.RX,
            "frequencies": [2],
            "input": [0],
            "warning": False,
        },
        {
            "encoding_unitary": [Gates.RX, Gates.RY],
            "frequencies": [2, 2],
            "input": [[0, 0]],
            "warning": False,
        },
        {"encoding_unitary": "RX", "frequencies": [2], "input": [0], "warning": False},
        {
            "encoding_unitary": ["RX", "RY"],
            "frequencies": [2, 2],
            "input": [[0, 0]],
            "warning": False,
        },
        {
            "encoding_unitary": ["RX", "RY"],
            "frequencies": [2, 2],
            "input": [0],
            "warning": True,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            encoding=test_case["encoding_unitary"],
            remove_zero_encoding=False,
        )

        if test_case["warning"]:
            with pytest.warns(UserWarning):
                _ = model(
                    model.params,
                    inputs=test_case["input"],
                )
        else:
            _ = model(
                model.params,
                inputs=test_case["input"],
            )
        assert (
            model.frequencies == test_case["frequencies"]
        ), f"Frequencies is not correct: got {model.frequencies},\
            expected {test_case['frequencies']}"


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
                "enc_params": model.enc_params,
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
    else:
        raise Exception("Cache file does not exist.")

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
def test_basic_draw() -> None:
    for ansatz in Ansaetze.get_available():
        # for ansatz in [Ansaetze.Circuit_9]:
        # No inputs
        model = Model(
            n_qubits=4,
            n_layers=1,
            circuit_type=ansatz.__name__,
            initialization="random",
            output_qubit=-1,
            remove_zero_encoding=False,
        )

        if model.params.size >= 4:
            rest_pi = int((model.params.size - 4) / 2)
            rest = int(model.params.size - rest_pi - 4)

            test_params = np.array(
                [
                    np.pi,  # Exactly pi
                    0,  # Zero
                    2 * np.pi,  # denominator=1
                    np.pi / 2,  # numerator=1
                ]
                + [
                    random.randint(1, 24) * np.pi / random.randint(1, 12)
                    for _ in range(rest_pi)
                ]
                + [np.random.uniform(0, 2 * np.pi) for _ in range(rest)]
            ).reshape(model.params.shape)
            model.params = test_params
        repr(model)
        _ = model.draw(figure="mpl")
        _ = model.draw(figure="tikz")


@pytest.mark.smoketest
def test_advanced_draw() -> None:
    model = Model(
        n_qubits=4,
        n_layers=1,
        circuit_type="Circuit_19",
        initialization="random",
        output_qubit=0,
        encoding=["RX", "RY"],
        remove_zero_encoding=False,
    )

    if model.params.size >= 4:
        rest_pi = int((model.params.size - 4) / 2)
        rest = int(model.params.size - rest_pi - 4)

        test_params = np.array(
            [
                np.pi,  # Exactly pi
                0,  # Zero
                2 * np.pi,  # denominator=1
                np.pi / 2,  # numerator=1
            ]
            + [
                random.randint(1, 24) * np.pi / random.randint(1, 12)
                for _ in range(rest_pi)
            ]
            + [np.random.uniform(0, 2 * np.pi) for _ in range(rest)]
        ).reshape(model.params.shape)
        model.params = test_params
    repr(model)
    _ = model.draw(figure="mpl")

    # No inputs and gate values
    quantikz_str = model.draw(figure="tikz", gate_values=True)
    quantikz_str.export("./tikz_test.tex", full_document=False, mode="w")

    # Inputs and gate values
    quantikz_str = model.draw(inputs=1.0, figure="tikz", gate_values=True)
    quantikz_str.export("./tikz_test.tex", full_document=False, mode="a")

    # No gate values, default input symbols
    quantikz_str = model.draw(figure="tikz", gate_values=False)
    quantikz_str.export("./tikz_test.tex", full_document=False, mode="a")

    # No gate values, custom input symbols
    quantikz_str = model.draw(
        figure="tikz", gate_values=False, inputs_symbols=["x", "y"]
    )
    quantikz_str.export("./tikz_test.tex", full_document=False, mode="a")


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
            "enc": Gates.RX,
            "dru": False,
            "degree": 1,
        },
        {
            "enc": Gates.RX,
            "dru": True,
            "degree": 4,
        },
        {
            "enc": Gates.RX,
            "dru": [[True, False], [False, True]],
            "degree": 2,
        },
        {
            "enc": [Gates.RX, Gates.RY],
            "dru": [[[0, 1], [1, 1]], [[1, 1], [0, 1]]],
            "degree": 4,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=2,
            encoding=test_case["enc"],
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
                "MultiQubitDepolarizing": 0.6,
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
