import pytest
import jax
import jax.numpy as jnp

from qml_essentials.yaqsi import (
    QuantumScript,
    H,
    CX,
    CCX,
    RX,
    PauliX,
    PauliZ,
    Hermitian,
    evolve,
    Z,
)

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def bell_circuit(*args, **kwargs):
    H(wires=0)
    CX(wires=[0, 1])


def parameterized_circuit(theta):
    RX(theta, wires=0)


def evol_circuit(t):
    time_evol = evolve(Hermitian(matrix=Z, wires=0))
    time_evol(t=t, wires=0)


def evol_circuit_plus(t):
    H(wires=0)  # prepare |+⟩
    time_evol = evolve(Hermitian(matrix=Z, wires=0))
    time_evol(t=t, wires=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_expval_bell_x() -> None:
    """Bell state (|00⟩+|11⟩)/√2 has ⟨Xᵢ⟩ = 0 (no single-qubit X bias)."""
    script = QuantumScript(f=bell_circuit)
    res = script.execute(type="expval", obs=[PauliX(0), PauliX(1)])
    assert jnp.allclose(res, jnp.array([0.0, 0.0]), atol=1e-10)


@pytest.mark.unittest
def test_probs_bell() -> None:
    """Bell state yields |00⟩ and |11⟩ each with probability 0.5."""
    script = QuantumScript(f=bell_circuit)
    probs = script.execute(type="probs")
    assert jnp.allclose(probs, jnp.array([0.5, 0.0, 0.0, 0.5]), atol=1e-10)


@pytest.mark.unittest
def test_expval_bell_z() -> None:
    """Bell state has ⟨Zᵢ⟩ = 0 for each qubit individually."""
    script = QuantumScript(f=bell_circuit)
    res_zz = script.execute(type="expval", obs=[PauliZ(0), PauliZ(1)])
    assert jnp.allclose(res_zz, jnp.array([0.0, 0.0]), atol=1e-10)


@pytest.mark.unittest
def test_parameterized_expval() -> None:
    """RX(θ)|0⟩ gives ⟨Z⟩ = cos(θ)."""
    script = QuantumScript(f=parameterized_circuit)
    theta_val = jnp.array(0.5)

    def cost(theta):
        return script.execute(type="expval", obs=[PauliZ(0)], args=(theta,))[0]

    assert jnp.allclose(cost(theta_val), jnp.cos(theta_val), atol=1e-6)


@pytest.mark.unittest
def test_jax_gradient() -> None:
    """Gradient of ⟨Z⟩ w.r.t. θ for RX(θ)|0⟩ equals -sin(θ)."""
    script = QuantumScript(f=parameterized_circuit)
    theta_val = jnp.array(0.5)

    def cost(theta):
        return script.execute(type="expval", obs=[PauliZ(0)], args=(theta,))[0]

    grad_fn = jax.grad(cost)
    expected_grad = -jnp.sin(theta_val)
    assert jnp.allclose(grad_fn(theta_val), expected_grad, atol=1e-6)


@pytest.mark.unittest
def test_evolve_from_zero() -> None:
    """exp(-i·t·Z)|0⟩ leaves ⟨X⟩ = 0 since |0⟩ is a Z eigenstate."""
    script = QuantumScript(f=evol_circuit)
    res = script.execute(type="expval", obs=[PauliX(0)], args=(0.3,))
    assert jnp.allclose(res, jnp.array([0.0]), atol=1e-6)


@pytest.mark.unittest
def test_evolve_from_plus() -> None:
    """exp(-i·t·Z)|+⟩ gives |⟨X⟩| = cos(2t)."""
    t = 0.3
    script = QuantumScript(f=evol_circuit_plus)
    res = script.execute(type="expval", obs=[PauliX(0)], args=(t,))
    assert jnp.allclose(jnp.abs(res), jnp.cos(2 * t), atol=1e-6)


# ---------------------------------------------------------------------------
# GHZ state tests  (n-qubit, exercises arbitrary-wire tensor contraction)
# ---------------------------------------------------------------------------


def ghz_circuit_3(*args, **kwargs):
    """3-qubit GHZ: (|000⟩ + |111⟩) / √2  using H + CNOT chain."""
    H(wires=0)
    CX(wires=[0, 1])
    CX(wires=[1, 2])


def ghz_circuit_4(*args, **kwargs):
    """4-qubit GHZ: (|0000⟩ + |1111⟩) / √2  using H + CNOT chain."""
    H(wires=0)
    CX(wires=[0, 1])
    CX(wires=[1, 2])
    CX(wires=[2, 3])


def ghz_toffoli_3(*args, **kwargs):
    """
    Alternative 3-qubit GHZ via a Toffoli gate.

    Start: |+⟩|0⟩|0⟩  (H on qubit 0)
    Then CCX(0,1→2) doesn't flip anything useful yet, so we first create
    |11⟩ on qubits 0,1 with probability 0.5, then Toffoli flips qubit 2.

    Circuit: H(0) → CX(0,1) → CCX(0,1,2)
    State after H+CX:  (|00⟩+|11⟩)/√2 ⊗ |0⟩
    After CCX (flips q2 only when q0=q1=1):  (|000⟩+|111⟩)/√2
    """
    H(wires=0)
    CX(wires=[0, 1])
    CCX(wires=[0, 1, 2])


@pytest.mark.unittest
def test_probs_ghz_3() -> None:
    """3-qubit GHZ state has prob 0.5 on |000⟩ and |111⟩, zero elsewhere."""
    script = QuantumScript(f=ghz_circuit_3)
    probs = script.execute(type="probs")
    expected = jnp.zeros(8).at[0].set(0.5).at[7].set(0.5)
    assert jnp.allclose(probs, expected, atol=1e-10)


@pytest.mark.unittest
def test_expval_ghz_3_z() -> None:
    """Each qubit of a 3-qubit GHZ state has ⟨Zᵢ⟩ = 0."""
    script = QuantumScript(f=ghz_circuit_3)
    obs = [PauliZ(0), PauliZ(1), PauliZ(2)]
    res = script.execute(type="expval", obs=obs)
    assert jnp.allclose(res, jnp.zeros(3), atol=1e-10)


@pytest.mark.unittest
def test_probs_ghz_4() -> None:
    """4-qubit GHZ state has prob 0.5 on |0000⟩ and |1111⟩, zero elsewhere."""
    script = QuantumScript(f=ghz_circuit_4)
    probs = script.execute(type="probs")
    expected = jnp.zeros(16).at[0].set(0.5).at[15].set(0.5)
    assert jnp.allclose(probs, expected, atol=1e-10)


@pytest.mark.unittest
def test_probs_ghz_toffoli() -> None:
    """3-qubit GHZ via Toffoli matches the CNOT-chain result."""
    script_cnot = QuantumScript(f=ghz_circuit_3)
    script_toff = QuantumScript(f=ghz_toffoli_3)
    probs_cnot = script_cnot.execute(type="probs")
    probs_toff = script_toff.execute(type="probs")
    assert jnp.allclose(probs_cnot, probs_toff, atol=1e-10)


@pytest.mark.unittest
def test_state_ghz_3_non_adjacent_wires() -> None:
    """CX on non-adjacent wires [0,2] still produces the correct entangled state."""

    def circuit(*args, **kwargs):
        H(wires=0)
        CX(wires=[0, 2])  # control=0, target=2 (skip qubit 1)

    script = QuantumScript(f=circuit)
    probs = script.execute(type="probs")
    # |000⟩ (index 0) and |101⟩ (index 5 = 0b101) each with prob 0.5
    expected = jnp.zeros(8).at[0].set(0.5).at[5].set(0.5)
    assert jnp.allclose(probs, expected, atol=1e-10)
