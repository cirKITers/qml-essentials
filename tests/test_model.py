import jax
from jax import random, grad, numpy as jnp
from typing import Any, Dict
import numpy as np
import random as pyrandom
import subprocess
import sys
import optax
from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze, Gates, Encoding
from qml_essentials.utils import PauliCircuit
from qml_essentials import simulation
from qml_essentials.pulses import PulseInformation
from qml_essentials.coefficients import Datasets
import pytest
import logging
import warnings
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

    assert not jnp.allclose(enc_params_before, enc_params_after), (
        "enc_params did not update during training"
    )

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
    model.transform_input = lambda inputs, enc_params: jnp.arccos(inputs)

    result_new = model(model.params, x, pulse_params=None)

    assert jnp.allclose(x, result_new), (
        "model.transform_input does not work as intended"
    )


@pytest.mark.unittest
def test_batching() -> None:
    for ansatz in Ansaetze.get_available(parameterized_only=True):
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type=ansatz.__name__,
        )
        print(ansatz.__name__)
        n_samples = 3
        model.initialize_params(random.key(1000), repeat=n_samples)
        params = model.params

    n_samples = 3
    model.initialize_params(random.key(1000), repeat=n_samples)
    params = model.params

    res = np.zeros((n_samples, 4, 4), dtype=jnp.complex128)
    for i in range(n_samples):
        res[i] = model(params=params[i], execution_type="density")

    assert res.shape == (n_samples, 4, 4), "Shape of batching is not correct"
    assert jnp.allclose(res, model(params=params, execution_type="density")), (
        "Content of batching is not equal"
    )


@pytest.mark.unittest
def test_repeat_batch_axis() -> None:
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        repeat_batch_axis=[False, True, True],
    )

    key = random.key(1000)
    key = model.initialize_params(key, repeat=10)
    res_a = model(inputs=random.uniform(key, (10, 1)))

    # we expect a batch size of 10 instead of 100
    assert res_a.shape == (
        10,
        2,
    ), f"Shape of repeat_batch_axis is not correct. Got {res_a.shape}"


@pytest.mark.unittest
def test_multiprocessing_expval() -> None:
    n_samples = 40000  # expval requires more samples for advantage

    model = Model(
        n_qubits=6,  # .. and larger circuits
        n_layers=6,
        circuit_type="Circuit_19",
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
    assert res_parallel.shape == res_single.shape, (
        "Shape of multiprocessing is not correct"
    )
    assert (res_parallel == res_single).all(), "Content of multiprocessing is not equal"


@pytest.mark.unittest
def test_random_key() -> None:
    model = Model(n_qubits=2, n_layers=1, circuit_type="Circuit_19", random_seed=1000)

    key_a = model.random_key
    key_b = model.initialize_params(key_a, repeat=10)

    assert key_a != key_b, "Keys should be different"
    assert key_b != model.random_key, "Keys should be different"


@pytest.mark.unittest
def test_random_key_call() -> None:
    """The internal key advances on eager calls, but a jitted call is traced
    once and replays the trace-time key, so an explicit ``random_key`` is the
    only way to get fresh randomness inside a trace."""
    kwargs: Dict[str, Any] = dict(
        inputs=jnp.array([0.0]),
        noise_params={"GateError": 0.3},
        execution_type="expval",
        force_mean=True,
    )

    def mk() -> Model:
        return Model(
            n_qubits=2, n_layers=1, circuit_type="Circuit_19", random_seed=1000
        )

    # eager: internal key advances, so stochastic results differ per call
    model = mk()
    key_before = model.random_key
    a = model(**kwargs)
    assert key_before != model.random_key, "Key should advance on an eager call"
    assert not jnp.allclose(a, model(**kwargs)), "Eager noise should be resampled"

    # an explicit key reproduces what the internal key would have produced
    model = mk()
    b = model(random_key=key_before, **kwargs)
    assert jnp.allclose(a, b), "Explicit key should match the internal key result"
    assert key_before == model.random_key, "Explicit key must not advance the state"

    # under jit the internal key is frozen: identical results on every call
    model = mk()
    key_before = model.random_key
    frozen = jax.jit(lambda p: model(params=p, **kwargs))
    params = jnp.array(model.params)
    with pytest.warns(UserWarning, match="replays the same noise realization"):
        first = frozen(params)
    assert jnp.allclose(first, frozen(params)), (
        "Without an explicit key, a jitted call replays the trace-time key"
    )
    assert key_before == model.random_key, "Tracer must not be stashed on the model"

    # ... whereas an explicit key gives fresh randomness per call
    model = mk()
    fresh = jax.jit(lambda p, k: model(params=p, random_key=k, **kwargs))
    key = random.key(1000)
    results = []
    for _ in range(3):
        key, sub_key = random.split(key)
        results.append(fresh(params, sub_key))
    assert not jnp.allclose(results[0], results[1])
    assert not jnp.allclose(results[1], results[2])


@pytest.mark.unittest
def test_frozen_randomness_warning() -> None:
    """Stochastic execution without an explicit key warns under a transform."""

    def mk(**kwargs: Any) -> Model:
        return Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            random_seed=1000,
            **kwargs,
        )

    inputs = jnp.array([0.5])
    noise_params = {"GateError": 0.3}

    model = mk()
    with pytest.warns(UserWarning, match="replays the same noise realization"):
        jax.jit(lambda p: model(params=p, inputs=inputs, noise_params=noise_params))(
            model.params
        )

    # shots are stochastic as well
    model = mk(shots=100)
    with pytest.warns(UserWarning, match="replays the same noise realization"):
        jax.jit(lambda p: model(params=p, inputs=inputs))(model.params)

    # no warning eagerly, with an explicit key, or without any randomness
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)

        model = mk()
        model(inputs=inputs, noise_params=noise_params)

        model = mk()
        jax.jit(
            lambda p, k: model(
                params=p, inputs=inputs, noise_params=noise_params, random_key=k
            )
        )(model.params, random.key(0))

        model = mk()
        jax.jit(lambda p: model(params=p, inputs=inputs))(model.params)


@pytest.mark.unittest
def test_next_key() -> None:
    """``next_key`` advances the internal key and feeds a traced call."""
    model = Model(n_qubits=2, n_layers=1, circuit_type="Circuit_19", random_seed=1000)

    key_before = model.random_key
    first, second = model.next_key(), model.next_key()
    assert key_before != model.random_key, "next_key should advance the internal key"
    assert first != second, "next_key should return a fresh key each time"

    noisy = jax.jit(
        lambda p, k: model(
            params=p,
            inputs=jnp.array([0.5]),
            noise_params={"GateError": 0.3},
            random_key=k,
        )
    )
    params = jnp.array(model.params)
    assert not jnp.allclose(noisy(params, model.next_key()), noisy(params, first)), (
        "A key per call should give fresh noise inside the trace"
    )


@pytest.mark.unittest
def test_structural_change_invalidates_plan() -> None:
    """Changing the circuit structure must not reuse a cached batched plan."""

    def mk(**kwargs: Any) -> Model:
        return Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            random_seed=1000,
            **kwargs,
        )

    # batched inputs, so the cached (vmapped) execution path is used
    inputs = jnp.array([[0.5], [1.2]])

    model = mk()
    model(inputs=inputs)
    reused = model(inputs=inputs, data_reupload=False)
    assert jnp.allclose(reused, mk(data_reupload=False)(inputs=inputs)), (
        "data_reupload change must not reuse the previous plan"
    )

    model = mk()
    model(inputs=inputs)
    model.observables = 0
    assert jnp.allclose(model(inputs=inputs), mk(observables=0)(inputs=inputs)), (
        "observables change must not reuse the previous plan"
    )

    # noise is captured in the traced closure, and shot mode caches it too
    model = mk(shots=1000)
    key = random.key(1000)
    noiseless = model(inputs=inputs, random_key=key)
    noisy = model(inputs=inputs, noise_params={"BitFlip": 0.5}, random_key=key)
    assert not jnp.allclose(noiseless, noisy), (
        "noise change must not reuse the previous shot plan"
    )


@pytest.mark.unittest
def test_zero_inputs_eager_matches_traced() -> None:
    """Zero inputs must not take a circuit path that tracing cannot take."""
    kwargs: Dict[str, Any] = dict(
        inputs=jnp.array([0.0]),
        noise_params={"BitFlip": 0.2},
        random_key=random.key(1000),
    )

    def mk() -> Model:
        return Model(
            n_qubits=2, n_layers=1, circuit_type="Circuit_19", random_seed=1000
        )

    eager = mk()(**kwargs)
    model = mk()
    traced = jax.jit(lambda p: model(params=p, **kwargs))(jnp.array(model.params))
    assert jnp.allclose(eager, traced), (
        "Zero inputs with noise must give the same result eagerly and traced"
    )


@pytest.mark.unittest
def test_no_tracer_leak_on_model_state(caplog) -> None:
    """Traced arguments must not be stashed on the model, otherwise the next
    read of e.g. ``model.params`` raises an UnexpectedTracerError."""
    caplog.set_level(logging.DEBUG, logger="qml_essentials.model")
    model = Model(n_qubits=2, n_layers=1, circuit_type="Circuit_19", random_seed=1000)
    params = jnp.array(model.params)
    enc_params = jnp.array(model.enc_params)

    cost = jax.jit(
        lambda p, e: jnp.sum(
            model(
                params=p,
                enc_params=e,
                inputs=jnp.array([0.0]),
                execution_type="expval",
                force_mean=True,
            )
        )
    )
    cost(params, enc_params)

    for name in ("params", "enc_params"):
        assert not isinstance(getattr(model, name), jax.core.Tracer), (
            f"`{name}` must not hold a tracer after a traced call"
        )
        assert any(f"`{name}` is a JAX tracer" in r.message for r in caplog.records), (
            f"Skipping the `{name}` write should be reported at debug level"
        )

    # a second call reads the model state again; this raised before the guard
    cost(params, enc_params)
    grad(lambda p: jnp.sum(model(params=p, inputs=jnp.array([0.0]))))(params)


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
        {
            "encoding": Encoding("golomb", None),
            "degree": None,  # checked separately below
            "input": [0.5],
            "warning": False,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            encoding=test_case["encoding"],
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

        if test_case["degree"] is not None:
            assert model.degree == test_case["degree"], (
                f"Frequencies is not correct: got {model.degree},\
                expected {test_case['degree']} for test case {test_case}"
            )


@pytest.mark.unittest
def test_encoding_weights() -> None:
    """Encoding.get_weights returns the per-qubit weight vector w (phi_q = w_q x),
    consistent with the strategy scaling and with get_n_freqs."""
    from itertools import product

    for n in range(1, 6):
        cases = {
            "hamming": np.ones(n),
            "binary": 2.0 ** np.arange(n),
            "ternary": 3.0 ** np.arange(n),
        }
        for strategy, expected in cases.items():
            enc = Encoding(strategy, ["RX"])
            w = np.asarray(enc.get_weights(n))
            np.testing.assert_allclose(w, expected)
            # spectrum {sum_k s_k w_k : s_k in {-1,0,1}} has |Omega| = get_n_freqs(n)
            spectrum = {
                sum(s * wk for s, wk in zip(signs, w))
                for signs in product((-1, 0, 1), repeat=n)
            }
            n_freqs = enc.get_n_freqs(np.ones(n, dtype=bool))
            assert len(spectrum) == n_freqs, (
                f"{strategy}: |Omega|={len(spectrum)} != \
                get_n_freqs={n_freqs}"
            )

    with pytest.raises(ValueError):
        Encoding("golomb", None).get_weights(2)


@pytest.mark.unittest
@pytest.mark.parametrize("strategy", ["hamming", "binary", "ternary", "golomb"])
@pytest.mark.parametrize("n_qubits", [2, 3])
def test_encoding_spectrum_reference(strategy, n_qubits) -> None:
    """The FFT-significant frequencies of the model equal the spectrum Omega of
    Peters and Schuld (arXiv:2209.05523, Table 1) for each encoding strategy.
    For golomb, Omega is the sparse set of mark differences with
    |Omega| = d(d-1)+1; model.frequencies is its contiguous superset."""
    from qml_essentials.coefficients import Coefficients
    from qml_essentials.unitary import golomb_ruler

    n = n_qubits
    if strategy == "golomb":
        marks = golomb_ruler(2**n)
        expected = {a - b for a in marks for b in marks}
        assert len(expected) == 2**n * (2**n - 1) + 1
    else:
        half = {"hamming": n, "binary": 2**n - 1, "ternary": (3**n - 1) // 2}[strategy]
        expected = set(range(-half, half + 1))

    model = Model(
        n_qubits=n,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        encoding=Encoding(strategy, None if strategy == "golomb" else ["RX"]),
    )
    naive = set(int(v) for v in model.frequencies[0])
    if strategy == "golomb":
        assert expected.issubset(naive)
    else:
        assert naive == expected

    # oversample (mfs=2) so frequencies beyond the predicted range are visible
    coeffs, freqs = Coefficients.get_spectrum(model, mfs=2, shift=True)
    coeffs = np.asarray(coeffs).ravel()
    freqs = np.asarray(freqs).ravel()
    # Golomb coefficients (|R(k)| = 1) can be ~1e-5 for random parameters; the
    # float32 FFT noise floor is ~1e-7, so 1e-6 separates the two.
    significant = {int(round(f)) for f, c in zip(freqs, coeffs) if abs(c) > 1e-6}
    assert significant == expected, (
        f"{strategy} n={n}: FFT support {sorted(significant)} != {sorted(expected)}"
    )


@pytest.mark.unittest
def test_golomb_encoding() -> None:
    """Test the Golomb encoding strategy end-to-end.

    Verifies that:
    - The model instantiates correctly with Golomb encoding.
    - The Golomb ruler has the correct properties (all pairwise diffs distinct).
    - The diagonal unitary is applied as a multi-qubit gate.
    - The output changes with different inputs (encoding is effective).
    - Batched execution works.
    - The model produces correct degree/frequencies for the encoding.
    - Circuit drawing works with Golomb encoding.
    - Density matrix execution works (tests apply_to_density path).

    Reference: Peters et al., arXiv:2209.05523, Sec. 3.1 and Appendix C.4.
    """
    from qml_essentials.unitary import golomb_ruler

    # --- Golomb ruler validity ---
    for n_qubits in [1, 2, 3, 4]:
        d = 2**n_qubits
        marks = golomb_ruler(d)
        assert len(marks) == d, f"Expected {d} marks, got {len(marks)}"
        # All pairwise differences must be distinct
        diffs = []
        for i in range(len(marks)):
            for j in range(i + 1, len(marks)):
                diffs.append(marks[j] - marks[i])
        assert len(set(diffs)) == len(diffs), (
            f"Golomb ruler for d={d} has duplicate pairwise differences: marks={marks}"
        )

    # --- Model with Golomb encoding ---
    enc = Encoding("golomb", None)
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
        encoding=enc,
    )

    assert model.n_input_feat == 1, "Golomb encoding should have 1 input feature"
    assert enc.is_golomb, "Encoding should be identified as Golomb"

    # --- Forward pass produces valid output ---
    result_a = model(inputs=0.5)
    assert result_a.shape == (2,), f"Expected shape (2,), got {result_a.shape}"
    assert jnp.all(jnp.isfinite(result_a)), "Output contains non-finite values"

    # --- Different inputs produce different outputs ---
    result_b = model(inputs=1.5)
    assert not jnp.allclose(result_a, result_b), (
        "Different inputs should produce different outputs with Golomb encoding"
    )

    # --- Batched execution ---
    batch_inputs = jnp.array([0.1, 0.5, 1.0, 2.0])
    result_batch = model(inputs=batch_inputs)
    assert result_batch.shape == (
        4,
        2,
    ), f"Expected batch shape (4, 2), got {result_batch.shape}"

    # --- Degree / frequencies are consistent ---
    d = 2**model.n_qubits
    marks = golomb_ruler(d)
    max_mark = max(marks)
    # Golomb applies one multi-qubit diagonal gate per active layer
    n_app = int(np.count_nonzero(np.asarray(model.data_reupload[..., 0]).any(axis=1)))
    expected_n_freqs = 2 * n_app * max_mark + 1
    assert model.degree[0] == expected_n_freqs, (
        f"Expected degree {expected_n_freqs}, got {model.degree[0]}"
    )

    # --- Circuit drawing works ---
    text_repr = model.draw(inputs=0.5, figure="text")
    assert "DiagU" in text_repr, (
        "Circuit drawing should show DiagU gate for Golomb encoding"
    )

    # --- Density matrix execution ---
    rho = model(inputs=0.5, execution_type="density")
    assert rho.shape == (4, 4), f"Expected density shape (4, 4), got {rho.shape}"
    # Density matrix should be valid (trace ~1, Hermitian)
    assert jnp.isclose(jnp.trace(rho), 1.0, atol=1e-6), "Trace of rho should be 1"
    assert jnp.allclose(rho, rho.conj().T, atol=1e-6), "rho should be Hermitian"

    # --- Multiple layers with data reuploading ---
    model_dru = Model(
        n_qubits=2,
        n_layers=2,
        circuit_type="Circuit_1",
        encoding=enc,
        data_reupload=True,
    )
    result_dru = model_dru(inputs=0.5)
    assert jnp.all(jnp.isfinite(result_dru)), (
        "Multi-layer Golomb model output contains non-finite values"
    )

    # --- No data reuploading ---
    model_no_dru = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
        encoding=Encoding("golomb", None),
        data_reupload=False,
    )
    result_no_dru = model_no_dru(inputs=0.5)
    assert jnp.all(jnp.isfinite(result_no_dru)), (
        "No-DRU Golomb model output contains non-finite values"
    )

    # --- Invalid strategy raises error ---
    with pytest.raises(ValueError):
        Encoding("invalid_strategy", None)


@pytest.mark.unittest
def test_golomb_diagonal_decompose() -> None:
    """DiagonalQubitUnitary.decompose() reproduces exp(-i diag(marks) x) as a
    product of commuting Pauli-Z rotations (up to a global phase).

    Uses real marks (not the wrapped complex diagonal), so it stays exact even
    when x * max(mark) exceeds pi (n=3 has marks up to 44).
    """
    from functools import reduce
    from qml_essentials.unitary import golomb_ruler
    from qml_essentials.operations import DiagonalQubitUnitary, PauliRot

    for n in [1, 2, 3]:
        d = 2**n
        marks = jnp.array(golomb_ruler(d), dtype=float)
        w = 0.7
        diag = jnp.exp(-1j * marks * w)
        gate = DiagonalQubitUnitary(
            diag, wires=list(range(n)), generator=marks, scale=w, record=False
        )

        ops = gate.decompose()
        assert len(ops) >= 1, f"no factors for n={n}"
        for o in ops:
            assert isinstance(o, PauliRot)
            # Only diagonal (I/Z) strings, never the dropped identity.
            assert set(o.pauli_word) <= {"I", "Z"} and "Z" in o.pauli_word

        u_dec = reduce(lambda a, b: a @ b, [o.matrix for o in ops])
        expected = jnp.diag(diag)
        # Compare up to a global phase by normalising the (0, 0) entry.
        u_dec = u_dec / u_dec[0, 0]
        expected = expected / expected[0, 0]
        assert jnp.allclose(u_dec, expected, atol=1e-10), (
            f"decompose() mismatch for n={n}"
        )

    # A generic diagonal unitary without a stored generator is primitive.
    plain = DiagonalQubitUnitary(jnp.array([1.0, 1j]), wires=0, record=False)
    with pytest.raises(NotImplementedError):
        plain.decompose()


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
            observables=-1,
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
        observables=0,
        encoding=["RX", "RY"],
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
            observables=0,
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
    test_cases = [0.0, jnp.zeros(5), jnp.arange(5)]

    for inputs in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
        )

        _ = model(
            model.params,
            inputs=inputs,
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

    assert not jnp.allclose(model.params, temp_params, atol=1e-3), (
        "Re-Initialization failed!"
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

    assert not jnp.allclose(pulse_params_before, pulse_params_after), (
        "pulse_params did not update during training"
    )

    assert jnp.any(jnp.abs(grads[1]) > 1e-6), "Gradient wrt pulse_params is too small"


@pytest.mark.unittest
def test_pulse_model_inference():
    model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Hardware_Efficient",
    )

    inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

    # forward pass with initial pulse_params
    y_hat_original = model(
        inputs=inputs, pulse_params=model.pulse_params, force_mean=True
    )

    y_hat_unitary = model(inputs=inputs, force_mean=True)

    assert jnp.allclose(y_hat_unitary, y_hat_original, atol=1e-2), (
        "Unitary output did not match pulse output"
    )

    # perturb pulse_params
    original_params = model.pulse_params.copy()
    model.pulse_params += 0.1

    # forward pass with perturbed pulse_params
    y_hat_perturbed = model(
        inputs=inputs, pulse_params=model.pulse_params, force_mean=True
    )

    assert y_hat_original.shape[0] == inputs.shape[0], "Output batch size mismatch"

    # ensure output changed after perturbing pulse_params
    assert not jnp.allclose(y_hat_original, y_hat_perturbed), (
        "Pulse output did not change after modifying pulse_params"
    )

    model.pulse_params = original_params


@pytest.mark.unittest
def test_pulse_model_batching():
    random_key = random.key(1000)

    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")

    # test pulse params batching
    res_b = model(
        pulse_params=jnp.repeat(model.pulse_params, 2, axis=0),
    )

    # two qubits -> two expvals with batch size 2
    assert res_b.shape == (2, 2), "Batch size mismatch"

    inputs = random.uniform(random_key, (3,), maxval=2 * jnp.pi)
    random_key, _ = random.split(random_key)

    # test pulse params & inputs batching
    res_a = model(inputs=inputs)
    res_b = model(inputs=inputs, pulse_params=model.pulse_params)

    assert np.allclose(res_a.shape, res_b.shape), "Batch shape mismatch"
    assert jnp.allclose(res_a, res_b, atol=1e-2), (
        "Inputs batching failed. Results differ."
    )

    model.initialize_params(random_key, repeat=2)

    # test pulse params & params & inputs batching
    res_a = model(inputs=inputs)
    res_b = model(inputs=inputs, pulse_params=model.pulse_params)

    assert np.allclose(res_a.shape, res_b.shape), "Batch shape mismatch"
    assert jnp.allclose(res_a, res_b, atol=1e-2), (
        "Params batching failed. Results differ."
    )


@pytest.mark.unittest
def test_pulse_encoding_shape() -> None:
    model = Model(n_qubits=3, n_layers=2, circuit_type="Hardware_Efficient")

    # one scaler per encoding-gate pulse parameter, per qubit, per feature
    s = PulseInformation.gate_by_name("RX").size
    assert model.enc_pulse_params.shape == (1, model.n_layers, model.n_qubits, s), (
        "enc_pulse_params has unexpected shape"
    )

    # scalers are initialized to ones (no deviation from calibrated defaults)
    assert jnp.allclose(model.enc_pulse_params, 1.0), (
        "enc_pulse_params should be initialized to ones"
    )


@pytest.mark.unittest
def test_pulse_encoding_equivalence() -> None:
    model = Model(n_qubits=3, n_layers=1, circuit_type="Hardware_Efficient")

    # nonzero inputs are required: zero inputs skip encoding entirely
    inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

    y_unitary = model(inputs=inputs, force_mean=True)
    y_all_pulse = model(
        inputs=inputs,
        pulse_params=model.pulse_params,
        enc_pulse_params=model.enc_pulse_params,
        force_mean=True,
    )
    y_enc_pulse = model(
        inputs=inputs, enc_pulse_params=model.enc_pulse_params, force_mean=True
    )

    assert jnp.allclose(y_unitary, y_all_pulse, atol=1e-2), (
        "all_pulse output with unit scalers did not match unitary output"
    )

    assert jnp.allclose(y_unitary, y_enc_pulse, atol=1e-2), (
        "enc_pulse output with unit scalers did not match unitary output"
    )


@pytest.mark.unittest
def test_pulse_encoding_frequency_scaling() -> None:
    # Scaling the amplitude component of enc_pulse_params by s is the same
    # frequency-scaling knob as the unitary trainable frequency enc_params = s
    # (arXiv:2309.03279): under RWA the encoding gate implements RX/RY(s * x).
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
    inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

    scale = 1.5
    # amplitude entry of each encoding gate sits at its offset in the last axis
    amp_idx = jnp.array(model._enc_pulse_offsets)
    eta = model.enc_pulse_params.at[..., amp_idx].set(scale)

    # enc_pulse leaves enc_params at its default (ones), so evaluate it first
    y_enc_pulse = model(inputs=inputs, enc_pulse_params=eta, force_mean=True)
    y_unitary = model(
        inputs=inputs,
        enc_params=scale * jnp.ones_like(model.enc_params),
        force_mean=True,
    )

    assert jnp.allclose(y_enc_pulse, y_unitary, atol=1e-2), (
        "amplitude scaler did not act as the trainable-frequency knob enc_params"
    )


@pytest.mark.unittest
def test_pulse_encoding_effect() -> None:
    model = Model(n_qubits=3, n_layers=1, circuit_type="Hardware_Efficient")
    inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)

    def all_pulse(m):
        return m(
            inputs=inputs,
            pulse_params=m.pulse_params,
            enc_pulse_params=m.enc_pulse_params,
            force_mean=True,
        )

    def enc_pulse(m):
        return m(inputs=inputs, enc_pulse_params=m.enc_pulse_params, force_mean=True)

    def ansatz_pulse(m):
        return m(inputs=inputs, pulse_params=m.pulse_params, force_mean=True)

    y_all_pulse = all_pulse(model)
    y_enc_pulse = enc_pulse(model)
    y_pulse = ansatz_pulse(model)

    original = model.enc_pulse_params.copy()
    model.enc_pulse_params = model.enc_pulse_params + 0.1

    # perturbing enc_pulse_params changes all_pulse output ...
    assert not jnp.allclose(y_all_pulse, all_pulse(model)), (
        "all_pulse output did not change after perturbing enc_pulse_params"
    )

    # ... and equally the enc_pulse output ...
    assert not jnp.allclose(y_enc_pulse, enc_pulse(model)), (
        "enc_pulse output did not change after perturbing enc_pulse_params"
    )

    # ... but leaves the ansatz_pulse output (unitary encoding) untouched
    assert jnp.allclose(y_pulse, ansatz_pulse(model)), (
        "ansatz_pulse output changed after perturbing enc_pulse_params"
    )

    model.enc_pulse_params = original


@pytest.mark.unittest
def test_pulse_encoding_gradient() -> None:
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")

    domain = np.array([-jnp.pi, jnp.pi])
    x = jnp.linspace(domain[0], domain[1], num=10)
    y = jnp.stack([jnp.cos(sample) for sample in x])

    def cost_fct(enc_pp):
        y_hat = model(
            inputs=x,
            enc_pulse_params=enc_pp,
            pulse_params=model.pulse_params,
            force_mean=True,
        )
        return jnp.mean((y_hat - y) ** 2)

    enc_pp = model.enc_pulse_params.copy()
    grads = grad(cost_fct)(enc_pp)
    assert jnp.any(jnp.abs(grads) > 1e-6), "Gradient wrt enc_pulse_params is too small"

    # the study-4 conclusion rests on the ODE-adjoint gradient being correct:
    # verify it against a central finite difference on one amplitude coordinate
    idx = (0, 0, 0, model._enc_pulse_offsets[0])
    h = 1e-3
    fd = (cost_fct(enc_pp.at[idx].add(h)) - cost_fct(enc_pp.at[idx].add(-h))) / (2 * h)
    assert jnp.allclose(fd, grads[idx], rtol=5e-2, atol=1e-3), (
        f"autodiff grad {grads[idx]} disagrees with finite difference {fd}"
    )

    # use the original (non-traced) array for the optax step; reading back
    # model.enc_pulse_params after grad would return a leaked tracer (same
    # convention as pulse_params in test_pulse_model)
    opt = optax.adam(0.05)
    opt_state = opt.init(enc_pp)
    updates, opt_state = opt.update(grads, opt_state, enc_pp)
    enc_pp_after = optax.apply_updates(enc_pp, updates)

    assert not jnp.allclose(enc_pp, enc_pp_after), (
        "enc_pulse_params did not update during training"
    )


@pytest.mark.unittest
def test_pulse_encoding_batching() -> None:
    random_key = random.key(1000)
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")

    inputs = random.uniform(random_key, (3,), maxval=2 * jnp.pi)

    # batch enc_pulse_params along axis 0 (B_E = 2)
    batched = jnp.repeat(model.enc_pulse_params, 2, axis=0)
    res = model(inputs=inputs, enc_pulse_params=batched)

    # B_I = 3 inputs, B_E = 2 scaler sets, two qubits -> two expvals
    assert res.shape == (3, 2, 2), "enc_pulse_params batch shape mismatch"


@pytest.mark.unittest
def test_gate_mode_inference() -> None:
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
    inputs = jnp.linspace(-jnp.pi, jnp.pi, 5)

    # which pulse parameters are passed selects the execution mode
    cases = [
        ({}, "unitary"),
        ({"pulse_params": model.pulse_params}, "ansatz_pulse"),
        ({"enc_pulse_params": model.enc_pulse_params}, "enc_pulse"),
        (
            {
                "pulse_params": model.pulse_params,
                "enc_pulse_params": model.enc_pulse_params,
            },
            "all_pulse",
        ),
    ]
    for kwargs, mode in cases:
        y_inferred = model(inputs=inputs, force_mean=True, **kwargs)
        with pytest.warns(DeprecationWarning, match="gate_mode is deprecated"):
            y_legacy = model(inputs=inputs, force_mean=True, gate_mode=mode, **kwargs)

        assert jnp.allclose(y_inferred, y_legacy), (
            f"inferred mode does not match explicit gate_mode={mode}"
        )


@pytest.mark.unittest
def test_gate_mode_deprecated() -> None:
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")

    # an explicit gate_mode still works, but warns
    with pytest.warns(DeprecationWarning, match="gate_mode is deprecated"):
        model(gate_mode="unitary")

    # enc_pulse_params only allowed in the encoding-pulse modes
    for mode in ("unitary", "ansatz_pulse"):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="enc_pulse_params were provided"):
                model(enc_pulse_params=model.enc_pulse_params, gate_mode=mode)

    # pulse_params only allowed in the ansatz-pulse modes
    for mode in ("unitary", "enc_pulse"):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError, match="pulse_params were provided"):
                model(pulse_params=model.pulse_params, gate_mode=mode)

    # unknown gate_mode
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="Unknown gate_mode"):
            model(gate_mode="foobar")

    # pulse_params remain valid alongside all_pulse
    with pytest.warns(DeprecationWarning):
        model(pulse_params=model.pulse_params, gate_mode="all_pulse")

    # enc_pulse_params remain valid alongside enc_pulse
    with pytest.warns(DeprecationWarning):
        model(enc_pulse_params=model.enc_pulse_params, gate_mode="enc_pulse")


@pytest.mark.unittest
def test_pulse_encoding_errors() -> None:
    # golomb encoding has no pulse parametrization -> encoding pulses unsupported
    golomb_model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        encoding=Encoding("golomb", None),
    )

    # a custom encoding callable has no pulse parametrization either
    def custom_enc(inputs, wires, **kwargs):
        Gates.RX(inputs, wires=wires, **kwargs)

    custom_model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        encoding=custom_enc,
    )

    for incapable_model in (golomb_model, custom_model):
        with pytest.raises(ValueError, match="requires an encoding"):
            incapable_model(enc_pulse_params=incapable_model.enc_pulse_params)

        for mode in ("enc_pulse", "all_pulse"):
            with pytest.warns(DeprecationWarning):
                with pytest.raises(ValueError, match="requires an encoding"):
                    incapable_model(gate_mode=mode)


@pytest.mark.unittest
def test_draw_pulse() -> None:
    import matplotlib.pyplot as plt

    # nonzero inputs are required: zero inputs skip encoding entirely
    inputs = jnp.array([0.5])

    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
    fig, axes = model.draw_pulse(inputs=inputs)
    assert fig is not None, "draw_pulse did not return a figure"
    assert len(axes) == model.n_qubits, "one subplot per qubit expected"
    plt.close(fig)

    # an encoding without pulse parametrization falls back to the ansatz pulses
    golomb_model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        encoding=Encoding("golomb", None),
    )
    fig, _ = golomb_model.draw_pulse(inputs=inputs)
    plt.close(fig)

    # the mode selector is gone: everything with a pulse form is drawn
    with pytest.warns(DeprecationWarning, match="no longer takes gate_mode"):
        fig, _ = model.draw_pulse(inputs=inputs, gate_mode="ansatz_pulse")
    plt.close(fig)


@pytest.mark.unittest
def test_pulse_encoding_backward_compat() -> None:
    # a 3-element repeat_batch_axis (pre-enc_pulse_params) must still work
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Hardware_Efficient",
        repeat_batch_axis=[True, True, True],
    )
    inputs = jnp.linspace(-jnp.pi, jnp.pi, 5)

    # plain ansatz pulse mode is unaffected by the encoding-pulse extension
    res_u = model(inputs=inputs, force_mean=True)
    res_p = model(inputs=inputs, pulse_params=model.pulse_params, force_mean=True)
    assert jnp.allclose(res_u, res_p, atol=1e-2), "pulse output drifted from unitary"


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
            observables=0,
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
                assert inputs.shape[0] == 1, (
                    "expected one elemental input for zero dimensional output"
                )
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
            observables=0,
            shots=1024,
        )

        assert model.degree == test_case["degree"], (
            f"Expected frequencies {test_case['degree']} but got\
            {model.degree} for dru {test_case['dru']}"
        )

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
        observables=0,
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
            observables=0,
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
            observables=0,
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
            "observables": [0, 1],
            "shots": None,
            "force_mean": False,
            "out_shape": (2,),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "observables": [0, 1],
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "observables": [0, 1],
            "shots": None,
            "force_mean": True,
            "out_shape": (3,),
            "warning": False,
        },
        {
            "inputs": None,
            "execution_type": "density",
            "observables": -1,
            "shots": None,
            "force_mean": False,
            "out_shape": (4, 4),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "observables": -1,
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 4, 4),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "observables": 0,
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 2, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "observables": -1,
            "shots": 1024,
            "force_mean": False,
            "out_shape": (3, 2, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "observables": 0,
            "shots": 1024,
            "force_mean": False,
            "out_shape": (3, 2),
            "warning": False,
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "probs",
            "observables": [0, 1],
            "shots": 1024,
            "force_mean": True,
            "out_shape": (3, 2),
            "warning": False,
        },
        # {
        #     "inputs": jnp.array([0.1, 0.2, 0.3]),
        #     "execution_type": "probs",
        #     "observables": [0, 1],
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
            observables=test_case["observables"],
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

        assert out.shape == test_case["out_shape"], (
            f"Expected {test_case['out_shape']}, got shape {out.shape}\
            for test case {test_case}"
        )


@pytest.mark.unittest
def test_parity() -> None:
    model_a = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
        observables=[[0, 1]],  # parity
    )
    model_b = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
        observables=-1,  # individual
    )

    result_a = model_a(params=model_a.params, inputs=None, force_mean=True)
    result_b = model_b(
        params=model_a.params, inputs=None, force_mean=True
    )  # use same params!

    assert not jnp.allclose(result_a, result_b), (
        f"Models should be different! Got {result_a} and {result_b}"
    )


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
def test_training_with_data_reupload() -> None:
    """Test that jax.grad works when data_reupload is passed at call time."""
    model = Model(
        n_qubits=2,
        n_layers=2,
        circuit_type="Circuit_19",
        data_reupload=True,
    )

    params = model.params
    enc_params = model.enc_params
    # Simulate layerwise DRU: start with zeros (JAX array, as user code does)
    data_reupload = jnp.zeros(model.data_reupload.shape)
    domain_samples = jnp.linspace(-jnp.pi, jnp.pi, 5)
    fourier_samples = jnp.sin(domain_samples)

    def cost(params, inputs, targets, **kwargs):
        predictions = model(params=params, inputs=inputs, **kwargs)
        return jnp.mean((predictions - targets) ** 2)

    # This should not raise TracerBoolConversionError
    grads = grad(cost)(
        params,
        inputs=domain_samples,
        targets=fourier_samples,
        noise_params=None,
        enc_params=enc_params,
        data_reupload=data_reupload,
        execution_type="expval",
        force_mean=True,
    )

    assert grads.shape == params.shape, "Gradient shape mismatch"

    # Also test with a partially filled data_reupload
    data_reupload_np = np.zeros(model.data_reupload.shape)
    data_reupload_np[0, 0, 0] = 1
    grads2 = grad(cost)(
        params,
        inputs=domain_samples,
        targets=fourier_samples,
        noise_params=None,
        enc_params=enc_params,
        data_reupload=data_reupload_np,
        execution_type="expval",
        force_mean=True,
    )

    assert grads2.shape == params.shape, "Gradient shape mismatch (np array)"


@pytest.mark.unittest
def test_pauli_circuit_model() -> None:
    test_cases = [
        {
            "shots": None,
            "observables": 0,
            "inputs": jnp.array([0.5]),
        },
        {
            "shots": None,
            "observables": -1,
            "inputs": jnp.array([0.5]),
        },
        {
            "shots": None,
            "observables": 0,
            "inputs": None,
        },
        {
            "shots": None,
            "observables": -1,
            "inputs": None,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=3,
            n_layers=2,
            circuit_type="Circuit_19",
            observables=test_case["observables"],
            shots=test_case["shots"],
        )
        # Validate inputs for a single sample (not a batch)
        inputs = model._inputs_validation(test_case["inputs"])

        # Record the tape using jaqsi with a single input sample
        model_tape = model.script._record(
            params=model.params,
            inputs=inputs,
        )

        # Build observables from the model
        _, obs = model._build_obs()

        pauli_ops, pauli_obs = PauliCircuit.from_parameterised_circuit(model_tape, obs)

        result_circuit = model(
            model.params,
            inputs=test_case["inputs"],
        )

        # Execute the Pauli tape via jaqsi's statevector simulator
        result_pauli_circuit = simulation.simulate_and_measure(
            pauli_ops,
            model.n_qubits,
            "expval",
            pauli_obs,
            False,
        )

        assert all(
            jnp.isclose(result_circuit, result_pauli_circuit, atol=1e-5).flatten()
        ), (
            f"results of Pauli Circuit and basic Ansatz should be equal, but "
            f"are {result_pauli_circuit} and {result_circuit} for testcase "
            f"{test_case}, respectively."
        )


@pytest.mark.unittest
def test_exact_spectrum_subset_of_naive() -> None:
    """The exact FourierTree spectrum is a subset of the naive estimate and
    contains every frequency the FFT finds to be non-zero."""
    from qml_essentials.coefficients import Coefficients

    model = Model(n_qubits=3, n_layers=1, circuit_type="Circuit_19", observables=0)

    exact = model.exact_spectrum()
    assert len(exact) == model.n_input_feat

    naive = set(int(v) for v in model.frequencies[0])
    exact_set = set(int(v) for v in exact[0])
    assert exact_set.issubset(naive), (
        f"Exact spectrum {exact_set} not a subset of naive {naive}"
    )

    # The exact spectrum must match the FFT's significant frequencies.  A
    # magnitude threshold of 1e-4 separates genuine coefficients (~1e-1) from
    # the float32 FFT noise floor (~1e-7).
    fft_coeffs, fft_freqs = Coefficients.get_spectrum(
        model, force_mean=True, shift=True
    )
    fft_coeffs = np.asarray(fft_coeffs).ravel()
    fft_freqs = np.asarray(fft_freqs).ravel()
    significant = set(int(f) for f, c in zip(fft_freqs, fft_coeffs) if abs(c) > 1e-4)
    assert exact_set == significant, (
        f"Exact spectrum {exact_set} does not match FFT-significant "
        f"frequencies {significant}"
    )


@pytest.mark.unittest
def test_exact_spectrum_multi_feature() -> None:
    """The exact spectrum supports multiple input features, each a subset of
    the corresponding naive per-feature spectrum."""
    model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Circuit_19",
        observables=0,
        encoding=["RX", "RY"],
    )

    exact = model.exact_spectrum()
    assert len(exact) == 2

    for ex, naive in zip(exact, model.frequencies):
        assert set(int(v) for v in ex).issubset(set(int(v) for v in naive))
        assert 0 in set(int(v) for v in ex)  # DC present for this model


@pytest.mark.unittest
def test_exact_spectrum_symbolic_cancellation() -> None:
    """Identically-cancelling frequencies are excluded symbolically.

    The paper example (Wiedmann et al., Fig. 2): a feature encoded twice via
    RX on the same qubit with no variational gates in between combines into a
    single rotation of angle 2x, so <Z> = cos(2x).  The naive spectrum is
    {-2..2}; the exact spectrum is {-2, 2} (the omega=0 contributions of the
    cos^2 and sin^2 paths cancel identically)."""
    model = Model(
        n_qubits=1,
        n_layers=2,
        circuit_type="No_Ansatz",
        data_reupload=True,
        encoding="RX",
        observables=0,
    )

    assert set(int(v) for v in model.frequencies[0]) == {-2, -1, 0, 1, 2}

    exact = model.exact_spectrum()
    assert set(int(v) for v in exact[0]) == {-2, 2}, (
        f"Expected symbolic cancellation of omega=0, got {exact[0]}"
    )


@pytest.mark.smoketest
def test_gate_mode_training() -> None:
    model = Model(
        n_qubits=3,
        n_layers=1,
        circuit_type="Circuit_19",
    )

    domain_samples, fourier_samples, coefficients = Datasets.generate_fourier_series(
        random_key=model.random_key,
        model=model,
    )

    opt = optax.adam(0.001)
    params = model.params
    opt_state = opt.init(params)

    def cost(params, inputs, targets, **kwargs):
        y_hat = model(params=params, inputs=inputs, **kwargs)

        return jnp.mean((y_hat - targets) ** 2)

    start = time.time()
    for epoch in range(1, 1000):
        grads = grad(cost)(
            params,
            inputs=domain_samples,
            targets=fourier_samples,
            execution_type="expval",
            force_mean=True,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        model.params = params
    end = time.time()
    print(f"Time taken: {end - start}")
    assert end - start < 120, "Time limit of 120 seconds exceeded"


@pytest.mark.benchmark
@pytest.mark.unittest
def test_pulse_mode_training() -> None:
    original_rwa = PulseInformation.get_rwa()
    PulseInformation.set_rwa(True)

    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_1",
    )

    domain_samples, fourier_samples, coefficients = Datasets.generate_fourier_series(
        random_key=model.random_key,
        model=model,
    )

    opt = optax.adam(0.001)
    params = {"unitary": model.params, "pulse": model.pulse_params}
    opt_state = opt.init(params)

    def cost(params, inputs, targets, **kwargs):
        y_hat = model(
            params=params["unitary"],
            pulse_params=params["pulse"],
            inputs=inputs,
            **kwargs,
        )

        return jnp.mean((y_hat - targets) ** 2)

    start = time.time()
    for epoch in range(1, 5):
        grads = grad(cost)(
            params,
            inputs=domain_samples,
            targets=fourier_samples,
            execution_type="expval",
            force_mean=True,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        model.params = params["unitary"]
        model.pulse_params = params["pulse"]

    PulseInformation.set_rwa(original_rwa)
    end = time.time()
    print(f"Time taken: {end - start}")
    assert end - start < 60, "Time limit of 60 seconds exceeded"


@pytest.mark.unittest
def test_output_qubit_deprecated() -> None:
    """output_qubit is a deprecated alias that warns and forwards to observables."""
    inp = jnp.array([[0.5]])

    # constructor kwarg warns
    with pytest.warns(DeprecationWarning):
        m_old = Model(n_qubits=3, n_layers=1, random_seed=5, output_qubit=0)

    # forwarding equivalence: output_qubit=0 matches observables=0
    m_new = Model(n_qubits=3, n_layers=1, random_seed=5, observables=0)
    a = np.asarray(m_old(inputs=inp, execution_type="expval"))
    b = np.asarray(m_new(inputs=inp, execution_type="expval"))
    assert np.allclose(a, b)

    # property setter warns and forwards
    m2 = Model(n_qubits=3, n_layers=1, random_seed=5)
    with pytest.warns(DeprecationWarning):
        m2.output_qubit = 0
    assert m2._measured_wires == [0]

    # passing both raises
    with pytest.raises(ValueError):
        Model(n_qubits=3, n_layers=1, output_qubit=0, observables=0)


def _model_state(model: Model) -> dict:
    """Snapshot the per-call mutable state that ``Model.apply`` must not touch."""
    return {
        "params": np.array(model.params),
        "enc_params": np.array(model.enc_params),
        "pulse_params": np.array(model.pulse_params),
        "random_key": np.array(random.key_data(model.random_key)),
        "noise_params": model.noise_params,
        "execution_type": model.execution_type,
        "batch_shape": model._batch_shape,
    }


def _assert_state_unchanged(before: dict, after: dict) -> None:
    for key in ("params", "enc_params", "pulse_params", "random_key"):
        assert np.array_equal(before[key], after[key]), f"{key} was modified"
    for key in ("noise_params", "execution_type", "batch_shape"):
        assert before[key] == after[key], f"{key} was modified"


@pytest.mark.unittest
def test_import_without_matplotlib() -> None:
    """The core import path must not require matplotlib, which is dev-only."""
    script = (
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def guard(name, *args, **kwargs):\n"
        "    if name.split('.')[0] == 'matplotlib':\n"
        "        raise ImportError('matplotlib blocked')\n"
        "    return _real(name, *args, **kwargs)\n"
        "builtins.__import__ = guard\n"
        "from qml_essentials.model import Model\n"
        "model = Model(n_qubits=2, n_layers=1, circuit_type='Circuit_19')\n"
        "assert isinstance(model.draw(figure='text'), str)\n"
        "try:\n"
        "    model.draw(figure='mpl')\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected ImportError for figure=mpl')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.unittest
def test_apply_pure_under_jit() -> None:
    """An outer jit around apply must work and leave the model untouched."""
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        initialization="random",
    )

    before = _model_state(model)

    fn = jax.jit(lambda p, x: model.apply(params=p, inputs=x))

    out_a = fn(model.params, jnp.array([0.1, 0.2, 0.3]))
    # the second call is where a tracer leaked by the first one would surface
    out_b = fn(model.params, jnp.array([0.4, 0.5, 0.6]))

    assert out_a.shape == (3, 1, 1, 1, 2)
    assert out_b.shape == (3, 1, 1, 1, 2)
    assert not jnp.allclose(out_a, out_b)

    _assert_state_unchanged(before, _model_state(model))

    # a subsequent eager call is where a leaked tracer would raise
    assert model(inputs=jnp.array([0.4, 0.5, 0.6])).shape == (3, 2)


@pytest.mark.unittest
def test_apply_under_vmap() -> None:
    """apply must be vmap-able over per-sample inputs."""
    model = Model(n_qubits=2, n_layers=1, circuit_type="Circuit_19")

    inputs = jnp.array([[0.1], [0.2], [0.3]])
    before = _model_state(model)

    out = jax.vmap(lambda x: model.apply(inputs=x))(inputs)
    assert out.shape == (3, 1, 1, 1, 1, 2)

    ref = model.apply(inputs=inputs)
    assert ref.shape == (3, 1, 1, 1, 2)
    assert jnp.allclose(out.reshape(3, 2), ref.reshape(3, 2), atol=1e-6)

    _assert_state_unchanged(before, _model_state(model))


@pytest.mark.unittest
def test_apply_jit_grad_train_step() -> None:
    """A whole training step built on apply must be jit-able and differentiable."""
    model = Model(n_qubits=2, n_layers=1, circuit_type="Circuit_19")

    inputs = jnp.array([0.1, 0.2, 0.3])
    targets = jnp.array([0.1, -0.2, 0.3])

    def cost(params):
        y_hat = model.apply(params=params, inputs=inputs, force_mean=True)
        return jnp.mean((y_hat.reshape(-1) - targets) ** 2)

    step = jax.jit(jax.value_and_grad(cost))

    params = model.params
    loss_a, grads = step(params)
    params = params - 0.1 * grads
    loss_b, _ = step(params)

    assert jnp.isfinite(loss_a) and jnp.isfinite(loss_b)
    assert jnp.any(grads != 0.0), "expected non-zero gradients"


@pytest.mark.unittest
def test_apply_matches_call() -> None:
    """apply and __call__ agree numerically up to the squeeze."""
    inputs = jnp.array([0.1, 0.2, 0.3])

    for execution_type in ["expval", "density", "state"]:
        model = Model(n_qubits=2, n_layers=1, circuit_type="Circuit_19")

        out = model.apply(
            params=model.params, inputs=inputs, execution_type=execution_type
        )
        ref = model(model.params, inputs=inputs, execution_type=execution_type)

        assert jnp.allclose(jnp.squeeze(out), ref, atol=1e-6), (
            f"apply and __call__ disagree for execution_type={execution_type}"
        )


@pytest.mark.unittest
def test_apply_output_shape() -> None:
    """apply keeps a fixed rank, including for a single observable or batch of one."""
    test_cases = [
        {
            "inputs": jnp.array(0.1),
            "observables": [0, 1],
            "out_shape": (1, 1, 1, 1, 2),
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "observables": [0, 1],
            "out_shape": (3, 1, 1, 1, 2),
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "observables": 0,
            "out_shape": (3, 1, 1, 1, 1),
        },
        {
            "inputs": jnp.array(0.1),
            "observables": 0,
            "out_shape": (1, 1, 1, 1, 1),
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            observables=test_case["observables"],
        )
        out = model.apply(inputs=test_case["inputs"])

        assert out.shape == test_case["out_shape"], (
            f"Expected {test_case['out_shape']}, got shape {out.shape}\
            for test case {test_case}"
        )


@pytest.mark.unittest
def test_call_keepdims() -> None:
    """keepdims=True returns the full [B_I, B_P, B_R, B_E, O] shape."""
    test_cases = [
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "observables": [0, 1],
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 1, 1, 1, 2),
        },
        {
            "inputs": jnp.array(0.1),
            "execution_type": "expval",
            "observables": 0,
            "shots": None,
            "force_mean": False,
            "out_shape": (1, 1, 1, 1, 1),
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "expval",
            "observables": [0, 1],
            "shots": None,
            "force_mean": True,
            "out_shape": (3, 1, 1, 1, 1),
        },
        {
            "inputs": jnp.array([0.1, 0.2, 0.3]),
            "execution_type": "density",
            "observables": -1,
            "shots": None,
            "force_mean": False,
            "out_shape": (3, 1, 1, 1, 4, 4),
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            observables=test_case["observables"],
            shots=test_case["shots"],
        )
        out = model(
            model.params,
            inputs=test_case["inputs"],
            force_mean=test_case["force_mean"],
            noise_params=None,
            execution_type=test_case["execution_type"],
            keepdims=True,
        )

        assert out.shape == test_case["out_shape"], (
            f"Expected {test_case['out_shape']}, got shape {out.shape}\
            for test case {test_case}"
        )

    # the default stays fully squeezed
    model = Model(
        n_qubits=2,
        n_layers=1,
        circuit_type="Circuit_19",
        observables=[0, 1],
    )
    out = model(model.params, inputs=jnp.array([0.1, 0.2, 0.3]))
    assert out.shape == (3, 2)
