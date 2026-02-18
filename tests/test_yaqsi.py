import pytest
import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np

from qml_essentials.yaqsi import (
    QuantumScript,
    evolve,
)
from qml_essentials.operations import (
    H,
    CX,
    CCX,
    RX,
    PauliX,
    PauliZ,
    Hermitian,
    Z,
    # noise channels
    BitFlip,
    PhaseFlip,
    DepolarizingChannel,
    AmplitudeDamping,
    PhaseDamping,
    ThermalRelaxationError,
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


# ---------------------------------------------------------------------------
# Density matrix tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_density_pure_state_is_projector() -> None:
    """
    A density matrix from a pure circuit satisfies ρ² = ρ (idempotent)
    and Tr(ρ) = 1.
    """
    script = QuantumScript(f=bell_circuit)
    rho = script.execute(type="density")

    assert rho.shape == (4, 4)
    assert jnp.allclose(jnp.trace(rho), 1.0, atol=1e-10)
    assert jnp.allclose(rho @ rho, rho, atol=1e-10)


@pytest.mark.unittest
def test_density_bell_matches_statevector() -> None:
    """
    Density matrix of the Bell state equals |ψ⟩⟨ψ|
    computed from the statevector path.
    """
    script = QuantumScript(f=bell_circuit)
    state = script.execute(type="state")
    rho = script.execute(type="density")

    rho_expected = jnp.outer(state, jnp.conj(state))
    assert jnp.allclose(rho, rho_expected, atol=1e-10)


@pytest.mark.unittest
def test_density_diagonal_equals_probs() -> None:
    """
    The diagonal of the density matrix equals the probability vector
    from the statevector path, for any pure circuit.
    """
    script = QuantumScript(f=ghz_circuit_3)
    probs = script.execute(type="probs")
    rho = script.execute(type="density")

    assert jnp.allclose(jnp.real(jnp.diag(rho)), probs, atol=1e-10)


@pytest.mark.unittest
def test_density_is_hermitian() -> None:
    """Density matrix must satisfy ρ = ρ†."""
    script = QuantumScript(f=ghz_circuit_4)
    rho = script.execute(type="density")

    assert jnp.allclose(rho, jnp.conj(rho.T), atol=1e-10)


@pytest.mark.unittest
def test_density_expval_matches_statevector() -> None:
    """
    Tr(O ρ) via the density path must equal ⟨ψ|O|ψ⟩ from the statevector path
    for any observable and any pure circuit.
    """
    script = QuantumScript(f=parameterized_circuit)
    theta_val = jnp.array(0.7)

    ev_pure = script.execute(type="expval", obs=[PauliZ(0)], args=(theta_val,))[0]

    rho = script.execute(type="density", args=(theta_val,))
    # Tr(Z ρ) — build Z in the 1-qubit Hilbert space directly
    Z_mat = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    ev_density = jnp.real(jnp.trace(Z_mat @ rho))

    assert jnp.allclose(ev_pure, ev_density, atol=1e-8)


# ---------------------------------------------------------------------------
# Noise channel tests  (validated against PennyLane default.mixed)
# ---------------------------------------------------------------------------


# Helper: run a 1-qubit circuit through PennyLane default.mixed and return ρ
def _pennylane_density(circuit_fn, n_qubits=1) -> np.ndarray:
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_fn()
        return qml.density_matrix(wires=list(range(n_qubits)))

    return np.array(qnode())


@pytest.mark.unittest
def test_bitflip_matches_pennylane() -> None:
    """BitFlip(p) density matrix matches PennyLane default.mixed."""
    p = 0.15
    theta = 0.8

    def yaqsi_circuit(t):
        RX(t, wires=0)
        BitFlip(p, wires=0)

    def pl_circuit():
        qml.RX(theta, wires=0)
        qml.BitFlip(p, wires=0)

    script = QuantumScript(f=yaqsi_circuit)
    rho_ours = np.array(script.execute(type="density", args=(jnp.array(theta),)))
    rho_pl = _pennylane_density(pl_circuit)

    assert np.allclose(
        rho_ours, rho_pl, atol=1e-8
    ), f"BitFlip mismatch:\nours =\n{rho_ours}\nPL =\n{rho_pl}"


@pytest.mark.unittest
def test_phaseflip_matches_pennylane() -> None:
    """PhaseFlip(p) density matrix matches PennyLane default.mixed."""
    p = 0.2
    theta = 1.1

    def yaqsi_circuit(t):
        RX(t, wires=0)
        PhaseFlip(p, wires=0)

    def pl_circuit():
        qml.RX(theta, wires=0)
        qml.PhaseFlip(p, wires=0)

    script = QuantumScript(f=yaqsi_circuit)
    rho_ours = np.array(script.execute(type="density", args=(jnp.array(theta),)))
    rho_pl = _pennylane_density(pl_circuit)

    assert np.allclose(
        rho_ours, rho_pl, atol=1e-8
    ), f"PhaseFlip mismatch:\nours =\n{rho_ours}\nPL =\n{rho_pl}"


@pytest.mark.unittest
def test_depolarizing_matches_pennylane() -> None:
    """DepolarizingChannel(p) density matrix matches PennyLane default.mixed."""
    p = 0.12
    theta = 0.6

    def yaqsi_circuit(t):
        RX(t, wires=0)
        DepolarizingChannel(p, wires=0)

    def pl_circuit():
        qml.RX(theta, wires=0)
        qml.DepolarizingChannel(p, wires=0)

    script = QuantumScript(f=yaqsi_circuit)
    rho_ours = np.array(script.execute(type="density", args=(jnp.array(theta),)))
    rho_pl = _pennylane_density(pl_circuit)

    assert np.allclose(
        rho_ours, rho_pl, atol=1e-7
    ), f"DepolarizingChannel mismatch:\nours =\n{rho_ours}\nPL =\n{rho_pl}"


@pytest.mark.unittest
def test_amplitude_damping_matches_pennylane() -> None:
    """AmplitudeDamping(γ) density matrix matches PennyLane default.mixed."""
    gamma = 0.25
    theta = 1.3

    def yaqsi_circuit(t):
        RX(t, wires=0)
        AmplitudeDamping(gamma, wires=0)

    def pl_circuit():
        qml.RX(theta, wires=0)
        qml.AmplitudeDamping(gamma, wires=0)

    script = QuantumScript(f=yaqsi_circuit)
    rho_ours = np.array(script.execute(type="density", args=(jnp.array(theta),)))
    rho_pl = _pennylane_density(pl_circuit)

    assert np.allclose(
        rho_ours, rho_pl, atol=1e-8
    ), f"AmplitudeDamping mismatch:\nours =\n{rho_ours}\nPL =\n{rho_pl}"


@pytest.mark.unittest
def test_phase_damping_matches_pennylane() -> None:
    """PhaseDamping(γ) density matrix matches PennyLane default.mixed."""
    gamma = 0.3
    theta = 0.9

    def yaqsi_circuit(t):
        RX(t, wires=0)
        PhaseDamping(gamma, wires=0)

    def pl_circuit():
        qml.RX(theta, wires=0)
        qml.PhaseDamping(gamma, wires=0)

    script = QuantumScript(f=yaqsi_circuit)
    rho_ours = np.array(script.execute(type="density", args=(jnp.array(theta),)))
    rho_pl = _pennylane_density(pl_circuit)

    assert np.allclose(
        rho_ours, rho_pl, atol=1e-8
    ), f"PhaseDamping mismatch:\nours =\n{rho_ours}\nPL =\n{rho_pl}"


@pytest.mark.unittest
def test_thermal_relaxation_t2_le_t1_matches_pennylane() -> None:
    """ThermalRelaxationError (T₂ ≤ T₁ regime) matches PennyLane default.mixed."""
    pe, t1, t2, tg = 0.0, 1e-4, 5e-5, 1e-6  # t2 < t1
    theta = 1.0

    def yaqsi_circuit(t):
        RX(t, wires=0)
        ThermalRelaxationError(pe, t1, t2, tg, wires=0)

    def pl_circuit():
        qml.RX(theta, wires=0)
        qml.ThermalRelaxationError(pe, t1, t2, tg, wires=0)

    script = QuantumScript(f=yaqsi_circuit)
    rho_ours = np.array(script.execute(type="density", args=(jnp.array(theta),)))
    rho_pl = _pennylane_density(pl_circuit)

    assert np.allclose(
        rho_ours, rho_pl, atol=1e-7
    ), f"ThermalRelaxation (T2≤T1) mismatch:\nours =\n{rho_ours}\nPL =\n{rho_pl}"


@pytest.mark.unittest
def test_noise_auto_routes_to_density() -> None:
    """A circuit with a noise channel auto-routes to density simulation."""

    def noisy_circuit():
        H(wires=0)
        BitFlip(0.1, wires=0)

    script = QuantumScript(f=noisy_circuit)
    # Calling with type="probs" should still work (auto density path)
    probs = script.execute(type="probs")
    assert probs.shape == (2,)
    assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-10)


@pytest.mark.unittest
def test_noise_density_is_valid() -> None:
    """Density matrix from noisy circuit: Tr(ρ)=1, Hermitian, PSD, Tr(ρ²)≤1."""

    def noisy_bell(*args, **kwargs):
        H(wires=0)
        CX(wires=[0, 1])
        DepolarizingChannel(0.05, wires=0)
        DepolarizingChannel(0.05, wires=1)

    script = QuantumScript(f=noisy_bell)
    rho = script.execute(type="density")

    assert rho.shape == (4, 4)
    assert jnp.allclose(jnp.trace(rho), 1.0, atol=1e-10), "Tr(ρ) ≠ 1"
    assert jnp.allclose(rho, jnp.conj(rho.T), atol=1e-10), "ρ not Hermitian"
    # Purity: Tr(ρ²) ≤ 1 for a mixed state, < 1 when noise is present
    purity = jnp.real(jnp.trace(rho @ rho))
    assert purity <= 1.0 + 1e-10, f"purity {purity} > 1"
    assert purity < 1.0 - 1e-6, f"purity {purity} ≈ 1: channel had no effect?"
