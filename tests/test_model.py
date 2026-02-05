from typing import Optional
from jax import random, grad, numpy as jnp
import numpy as np
import random as pyrandom
import optax
from qml_essentials.model import Model
from qml_essentials.ansaetze import Circuit, Ansaetze, Gates, Encoding
from qml_essentials.ansaetze import PulseInformation as pinfo
from qml_essentials.utils import PauliCircuit
import pytest
import inspect
import logging
import pennylane as qml
import time

logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_trainable_frequencies() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        trainable_frequencies=True,
    )

    # setting test data
    domain = jnp.array([-jnp.pi, jnp.pi])
    omegas = jnp.array([1.2, 2.6, 3.4, 4.9])
    coefficients = jnp.array([0.5, 0.5, 0.5, 0.5])
    n_d = int(jnp.ceil(2 * jnp.max(jnp.abs(domain)) * jnp.max(omegas)))
    x = jnp.linspace(domain[0], domain[1], num=n_d)

    def f(x):
        return (
            1 / jnp.linalg.norm(omegas) * jnp.sum(coefficients * jnp.cos(omegas.T * x))
        )

    y = jnp.stack([f(sample) for sample in x])

    def cost_fct(all_params):
        y_hat = model(
            params=all_params[0], enc_params=all_params[1], inputs=x, force_mean=True
        )
        return jnp.mean((y_hat - y) ** 2)

    enc_params_before = model.enc_params.copy()
    opt = optax.adam(0.01)
    all_params = (model.params, model.enc_params)

    opt_state = opt.init((all_params))

    grads = grad(cost_fct)(all_params)

    updates, opt_state = opt.update(grads, opt_state, all_params)
    model.params, model.enc_params = optax.apply_updates(all_params, updates)
    enc_params_after = model.enc_params.copy()

    assert not jnp.allclose(
        enc_params_before, enc_params_after
    ), "enc_params did not update during training"

    assert jnp.any(jnp.abs(grads[1]) > 1e-6), "Gradient wrt enc_params is too small"

    # Smoketest to check model outside training
    model(enc_params=jnp.array(model.enc_params))
    model.trainable_frequencies = False
    model(enc_params=jnp.array(model.enc_params))


@pytest.mark.unittest
def test_transform_input() -> None:
    domain = jnp.array([-1, 1])
    omegas = jnp.array([1, 2, 3, 4])
    n_d = int(jnp.ceil(2 * jnp.max(jnp.abs(domain)) * jnp.max(omegas)))
    x = jnp.linspace(domain[0], domain[1], num=n_d)

    model = Model(
        n_qubits=1,
        n_layers=1,
        circuit_type="No_Ansatz",
        encoding="RX",
        data_reupload=False,
    )

    # Test the intended use of transform_input()
    inputs = jnp.array([[0.5, -0.2]])
    enc_params = jnp.array([2.0, 3.0])

    # Test for qubit 0, feature 0
    result = model.transform_input(inputs, enc_params)
    expected = enc_params * inputs
    assert jnp.allclose(result, expected), "Incorrect transform for qubit 0"

    # Test modified transform_input()
    model.transform_input = lambda inputs, enc_params: (jnp.arccos(inputs))

    result_new = model(model.params, x, pulse_params=None)

    assert jnp.allclose(
        x, result_new
    ), "model.transform_input does not work as intended"


@pytest.mark.unittest
def test_batching() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    n_samples = 3
    model.initialize_params(random.key(1000), repeat=n_samples)
    params = model.params

    res = np.zeros((n_samples, 4, 4), dtype=jnp.complex128)
    for i in range(n_samples):
        res[i] = model(params=params[:, :, i], execution_type="density")

    assert res.shape == (n_samples, 4, 4), "Shape of batching is not correct"
    assert jnp.allclose(
        res, model(params=params, execution_type="density")
    ), "Content of batching is not equal"


@pytest.mark.unittest
def test_repeat_batch_axis() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        repeat_batch_axis=[False, True, True],
    )

    key = random.key(1000)
    key, _ = model.initialize_params(key, repeat=10)
    res_a = model(inputs=random.uniform(key, (10, 1)))

    # we expect a batch size of 10 instead of 100
    assert res_a.shape == (
        10,
        2,
    ), f"Shape of repeat_batch_axis is not correct. Got {res_a.shape}"


@pytest.mark.unittest
def test_multiprocessing_density() -> None:
    # use n_samples that is not a multiple of the threshold
    n_samples = 1000

    model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Circuit_19",
        use_multithreading=True,
    )

    model.initialize_params(random.key(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_parallel = model(params=params, execution_type="density")
    t_parallel = time.time() - start

    model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    model.initialize_params(random.key(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_single = model(params=params, execution_type="density")
    t_single = time.time() - start
    # assert (
    #     t_parallel < t_single
    # ), "Time required for multiprocessing larger than single process"
    print(f"Diff: {t_parallel - t_single}")

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
        use_multithreading=True,
    )

    model.initialize_params(random.key(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_parallel = model(params=params, execution_type="expval")
    t_parallel = time.time() - start

    model = Model(
        n_qubits=6,
        n_layers=6,
        circuit_type="Circuit_19",
    )

    model.initialize_params(random.key(1000), repeat=n_samples)
    params = model.params

    start = time.time()
    res_single = model(params=params, execution_type="expval")
    t_single = time.time() - start

    # assert (
    #     t_parallel < t_single
    # ), "Time required for multiprocessing larger than single process"

    print(f"Diff: {t_parallel - t_single}")
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
            "encoding": Gates.RX,
            "degree": (5,),
            "input": [0],
            "warning": False,
        },
        {
            "encoding": [Gates.RX, Gates.RY],
            "degree": (5, 5),
            "input": [[0, 0]],
            "warning": False,
        },
        {
            "encoding": ["RX", Gates.RY],
            "degree": (5, 5),
            "input": [[0, 0]],
            "warning": False,
        },
        {"encoding": "RX", "degree": (5,), "input": [0], "warning": False},
        {
            "encoding": ["RX", "RY"],
            "degree": (5, 5),
            "input": [[0, 0]],
            "warning": False,
        },
        {
            "encoding": ["RX", "RY"],
            "degree": (5, 5),
            "input": [0],
            "warning": True,
        },
        {
            "encoding": Encoding("binary", ["RX"]),
            "degree": (7,),
            "input": [0],
            "warning": False,
        },
        {
            "encoding": Encoding("ternary", ["RX"]),
            "degree": (9,),
            "input": [0],
            "warning": False,
        },
        {
            "encoding": Encoding("ternary", ["RX", "RY"]),
            "degree": (9, 9),
            "input": [[0, 0]],
            "warning": False,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            encoding=test_case["encoding"],
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
            model.degree == test_case["degree"]
        ), f"Frequencies is not correct: got {model.degree},\
            expected {test_case['degree']} for test case {test_case}"


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
                    jnp.pi,  # Exactly pi
                    0,  # Zero
                    2 * jnp.pi,  # denominator=1
                    jnp.pi / 2,  # numerator=1
                ]
                + [
                    pyrandom.randint(1, 24) * jnp.pi / pyrandom.randint(1, 12)
                    for _ in range(rest_pi)
                ]
                + [np.random.uniform(0, 2 * jnp.pi) for _ in range(rest)]
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
                jnp.pi,  # Exactly pi
                0,  # Zero
                2 * jnp.pi,  # denominator=1
                jnp.pi / 2,  # numerator=1
            ]
            + [
                pyrandom.randint(1, 24) * jnp.pi / pyrandom.randint(1, 12)
                for _ in range(rest_pi)
            ]
            + [np.random.uniform(0, 2 * jnp.pi) for _ in range(rest)]
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
            execution_type="expval",
        )


@pytest.mark.smoketest
def test_inputs() -> None:
    test_cases = [
        {"inputs": 0.0, "remove_zero_encoding": True},
        {"inputs": 0.0, "remove_zero_encoding": False},
        {"inputs": jnp.zeros(5), "remove_zero_encoding": True},
        {"inputs": jnp.zeros(5), "remove_zero_encoding": False},
        {"inputs": jnp.arange(5), "remove_zero_encoding": True},
        {"inputs": jnp.arange(5), "remove_zero_encoding": False},
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            remove_zero_encoding=test_case["remove_zero_encoding"],
        )

        _ = model(
            model.params,
            inputs=test_case["inputs"],
            noise_params=None,
            execution_type="expval",
        )


@pytest.mark.unittest
def test_re_initialization() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        initialization_domain=[-2 * jnp.pi, 0],
        random_seed=1000,
    )

    assert model.params.max() <= 0, "Parameters should be in [-2pi, 0]!"

    temp_params = model.params.copy()

    model.initialize_params(random.key(1001))

    assert not jnp.allclose(
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

            n_params += (n_qubits - 1) * pinfo.num_params("CZ")

            return n_params

        @staticmethod
        def get_control_indices(n_qubits: int) -> Optional[jnp.ndarray]:
            return None

        @staticmethod
        def build(w: jnp.ndarray, n_qubits: int, **kwargs):
            w_idx = 0
            for q in range(n_qubits):
                Gates.RY(w[w_idx], wires=q, **kwargs)
                w_idx += 1
                Gates.RZ(w[w_idx], wires=q, **kwargs)
                w_idx += 1

            for q in range(n_qubits - 1):
                Gates.CZ(wires=[q, q + 1], **kwargs)

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
def test_pulse_model() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Hardware_Efficient",
    )

    # setting test data
    domain = np.array([-jnp.pi, jnp.pi])
    omegas = jnp.array([1, 2, 3, 4])
    coefficients = jnp.array([1, 1, 1, 1])
    n_d = int(jnp.ceil(2 * jnp.max(jnp.abs(domain)) * jnp.max(omegas)))
    x = jnp.linspace(domain[0], domain[1], num=n_d)

    def f(x):
        return (
            1 / jnp.linalg.norm(omegas) * jnp.sum(coefficients * jnp.cos(omegas.T * x))
        )

    y = jnp.stack([f(sample) for sample in x])

    def cost_fct(all_params):
        y_hat = model(
            params=all_params[0],
            pulse_params=all_params[1],
            inputs=x,
            force_mean=True,
            gate_mode="pulse",
        )
        return jnp.mean((y_hat - y) ** 2)

    opt = optax.adam(0.01)
    pulse_params_before = model.pulse_params.copy()
    all_params = (model.params, model.pulse_params)
    opt_state = opt.init((all_params))

    grads = grad(cost_fct)(all_params)

    updates, opt_state = opt.update(grads, opt_state, all_params)
    model.params, model.pulse_params = optax.apply_updates(all_params, updates)

    pulse_params_after = model.pulse_params.copy()

    assert not jnp.allclose(
        pulse_params_before, pulse_params_after
    ), "pulse_params did not update during training"

    assert jnp.any(jnp.abs(grads[1]) > 1e-6), "Gradient wrt pulse_params is too small"


@pytest.mark.expensive
@pytest.mark.unittest
def test_pulse_model_inference():
    model = Model(
        n_qubits=4,
        n_layers=2,
        circuit_type="Hardware_Efficient",
    )

    inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

    # forward pass with initial pulse_params
    y_hat_original = model(inputs=inputs, gate_mode="pulse", force_mean=True)

    y_hat_unitary = model(inputs=inputs, gate_mode="unitary", force_mean=True)

    assert jnp.allclose(
        y_hat_unitary, y_hat_original, atol=1e-3
    ), "Unitary output did not match pulse output"

    # perturb pulse_params
    original_params = model.pulse_params.copy()
    model.pulse_params += 0.1

    # forward pass with perturbed pulse_params
    y_hat_perturbed = model(inputs=inputs, gate_mode="pulse", force_mean=True)

    assert y_hat_original.shape[0] == inputs.shape[0], "Output batch size mismatch"

    # ensure output changed after perturbing pulse_params
    assert not jnp.allclose(
        y_hat_original, y_hat_perturbed
    ), "Pulse output did not change after modifying pulse_params"

    model.pulse_params = original_params


@pytest.mark.unittest
def test_pulse_model_batching():
    random_key = random.key(1000)

    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")

    # test pulse params batching
    res_b = model(
        pulse_params=jnp.repeat(model.pulse_params, 2, axis=-1), gate_mode="pulse"
    )

    # two qubits -> two expvals with batch size 2
    assert res_b.shape == (2, 2), "Batch size mismatch"

    inputs = random.uniform(random_key, (3,), maxval=2 * jnp.pi)
    random_key, _ = random.split(random_key)

    # test pulse params & inputs batching
    res_a = model(inputs=inputs, gate_mode="unitary")
    res_b = model(inputs=inputs, gate_mode="pulse")

    assert np.allclose(res_a.shape, res_b.shape), "Batch shape mismatch"
    assert jnp.allclose(res_a, res_b, atol=1e-3), "Inputs batching failed!"

    model.initialize_params(random_key, repeat=2)

    # test pulse params & params & inputs batching
    res_a = model(inputs=inputs, gate_mode="unitary")
    res_b = model(inputs=inputs, gate_mode="pulse")

    assert np.allclose(res_a.shape, res_b.shape), "Batch shape mismatch"
    assert jnp.allclose(res_a, res_b, atol=1e-3), "Params batching failed!"


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
    input_cases = [2 * jnp.pi * i for i in input_cases]
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
            "degree": (3,),
        },
        {
            "enc": Gates.RX,
            "dru": True,
            "degree": (9,),
        },
        {
            "enc": Gates.RX,
            "dru": [[True, False], [False, True]],
            "degree": (5,),
        },
        {
            "enc": [Gates.RX, Gates.RY],
            "dru": [[[0, 1], [1, 1]], [[1, 1], [0, 1]]],
            "degree": (5, 9),
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
        ), f"Expected frequencies {test_case['degree']} but got\
            {model.degree} for dru {test_case['dru']}"

        _ = model(
            model.params,
            inputs=None,
            noise_params=None,
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
            noise_params=test_case["noise_params"],
            execution_type=test_case["execution_type"],
        )

        # check if setting in the forward call is working
        assert model.noise_params == test_case["noise_params"]
        assert model.execution_type == test_case["execution_type"]


@pytest.mark.unittest
def test_output_shapes() -> None:
    test_cases = [
        {
            "inputs": jnp.array(0.1),
            "execution_type": "expval",
            "output_qubit": [0, 1],
            "shots": None,
            "force_mean": False,
            "out_shape": (2,),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": [0, 1],
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "output_qubit": [0, 1],
            "shots": None,
            "force_mean": True,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "inputs": None,
            "execution_type": "density",
            "output_qubit": -1,
            "shots": None,
            "force_mean": False,
            "out_shape": (4, 4),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "output_qubit": -1,
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 4, 4),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "output_qubit": 0,
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 2, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "output_qubit": -1,
            "shots": 1024,
            "force_mean": False,
            "out_shape": (3, 2, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "output_qubit": 0,
            "shots": 1024,
            "force_mean": False,
            "out_shape": (3, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "output_qubit": [0, 1],
            "shots": 1024,
            "force_mean": True,
            "out_shape": (3, 2),
            "warning": False,
        },
        # {
        #     "inputs": jnp.array([0.1, 0.2, 0.3]),
        #     "execution_type": "probs",
        #     "output_qubit": [0, 1],
        #     "shots": 1024,
        #     "force_mean": False,
        #     "out_shape": (3, 2, 2),
        #     "warning": False,
        # },
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
                    force_mean=test_case["force_mean"],
                    noise_params=None,
                    execution_type=test_case["execution_type"],
                )
        else:
            out = model(
                model.params,
                inputs=test_case["inputs"],
                force_mean=test_case["force_mean"],
                noise_params=None,
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
        output_qubit=[[0, 1]],  # parity
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

    assert not jnp.allclose(
        result_a, result_b
    ), f"Models should be different! Got {result_a} and {result_b}"


@pytest.mark.smoketest
def test_training_step() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
    )
    opt = qml.AdamOptimizer(stepsize=0.01)

    def cost(params):
        return model(params=params, inputs=jnp.array([0]), force_mean=True)

    params, cost = opt.step_and_cost(cost, model.params)


@pytest.mark.unittest
def test_pauli_circuit_model() -> None:
    test_cases = [
        {
            "shots": None,
            "output_qubit": 0,
            "inputs": jnp.array([0.1, 0.2, 0.3]),
        },
        {
            "shots": None,
            "output_qubit": -1,
            "inputs": jnp.array([0.1, 0.2, 0.3]),
        },
        {
            "shots": None,
            "output_qubit": 0,
            "inputs": None,
        },
        {
            "shots": None,
            "output_qubit": -1,
            "inputs": None,
        },
    ]
    dev = qml.device("default.qubit", wires=3)

    for test_case in test_cases:
        model = Model(
            n_qubits=3,
            n_layers=2,
            circuit_type="Circuit_19",
            output_qubit=test_case["output_qubit"],
            shots=test_case["shots"],
        )
        inputs = model._inputs_validation(test_case["inputs"])
        model_tape = qml.workflow.construct_tape(model.circuit)(
            model.params,
            inputs=inputs,
        )
        pauli_tape = PauliCircuit.from_parameterised_circuit(model_tape)

        result_circuit = model(
            model.params,
            inputs=test_case["inputs"],
        )
        result_pauli_circuit = jnp.array(qml.execute([pauli_tape], dev)[0]).T

        assert all(
            jnp.isclose(result_circuit, result_pauli_circuit, atol=1e-5).flatten()
        ), (
            f"results of Pauli Circuit and basic Ansatz should be equal, but "
            f"are {result_pauli_circuit} and {result_circuit} for testcase "
            f"{test_case}, respectively."
        )
