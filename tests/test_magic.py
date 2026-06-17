from qml_essentials.model import Model
from qml_essentials.magic import Magic
from qml_essentials.operations import state_expectation

import logging
import math
from itertools import product

import pytest
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", True)

INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _oracle_m2(psi, n_qubits):
    # Independent brute-force M_2 via the NumPy PauliWord expectation.
    d = 2**n_qubits
    total = 0.0
    for combo in product("IXYZ", repeat=n_qubits):
        e = state_expectation("".join(combo), np.asarray(psi)).real
        total += e**4
    return -math.log(total / d)


def _random_state(seed, n_qubits):
    k1, k2 = jax.random.split(jax.random.key(seed))
    shape = (2**n_qubits,)
    v = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
    return v / jnp.linalg.norm(v)


@pytest.mark.unittest
def test_m2_stabilizer_states_are_zero():
    # Stabilizer states have zero magic.
    cases = {
        "|0>": (jnp.array([1, 0], dtype=complex), 1),
        "|+>": (jnp.array([INV_SQRT2, INV_SQRT2], dtype=complex), 1),
        "bell": (jnp.array([INV_SQRT2, 0, 0, INV_SQRT2], dtype=complex), 2),
    }
    for name, (psi, n) in cases.items():
        m2 = float(Magic._compute_m2(psi, n))
        assert abs(m2) < 1e-10, f"{name}: expected 0, got {m2}"


@pytest.mark.unittest
def test_m2_reference_value_t_plus():
    # T|+> = (|0> + e^{i pi/4} |1>) / sqrt(2); analytic M_2 = -ln(3/4).
    tplus = jnp.array([INV_SQRT2, INV_SQRT2 * np.exp(1j * np.pi / 4)], dtype=complex)
    m2 = float(Magic._compute_m2(tplus, 1))
    assert abs(m2 - (-math.log(0.75))) < 1e-10


@pytest.mark.unittest
@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_m2_matches_pauliword_oracle(n_qubits):
    psi = _random_state(2024 + n_qubits, n_qubits)
    kernel = float(Magic._compute_m2(psi, n_qubits))
    oracle = _oracle_m2(psi, n_qubits)
    assert abs(kernel - oracle) < 1e-9


@pytest.mark.unittest
def test_m2_purity_invariant():
    # (1/2^n) sum_P <P>^2 = Tr(rho^2) = 1 for any pure state.
    psi = _random_state(99, 3)
    exp = Magic._pauli_expectations(psi, 3)
    assert abs(float(jnp.sum(exp**2) / 8) - 1.0) < 1e-9


@pytest.mark.unittest
def test_sre_clifford_circuit_is_zero():
    model = Model(n_qubits=2, n_layers=1, circuit_type="No_Ansatz", data_reupload=False)
    m2 = float(Magic.stabilizer_renyi_entropy(model, n_samples=None))
    assert abs(m2) < 1e-10


@pytest.mark.unittest
@pytest.mark.parametrize("layers", [1, 2])
def test_non_clifford_count(layers):
    # Circuit_1 = RX + RZ blocks: 2 Pauli rotations per qubit per layer.
    model = Model(
        n_qubits=2, n_layers=layers, circuit_type="Circuit_1", data_reupload=False
    )
    assert Magic.non_clifford_count(model) == 2 * model.n_qubits * layers


@pytest.mark.smoketest
def test_sre_sampled_and_current_params():
    model = Model(n_qubits=3, n_layers=2, circuit_type="Circuit_1", data_reupload=False)
    sampled = float(
        Magic.stabilizer_renyi_entropy(
            model, n_samples=20, random_key=jax.random.key(1)
        )
    )
    current = float(Magic.stabilizer_renyi_entropy(model, n_samples=None))
    assert sampled >= -1e-9 and current >= -1e-9


@pytest.mark.smoketest
def test_m2_is_differentiable():
    def m2_of_theta(theta):
        psi = jnp.array([jnp.cos(theta), jnp.sin(theta) * jnp.exp(1j * np.pi / 4)])
        return Magic._compute_m2(psi, 1)

    grad = jax.grad(m2_of_theta)(0.5)
    assert jnp.isfinite(grad)
