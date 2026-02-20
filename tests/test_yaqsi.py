import pytest
import jax
import jax.numpy as jnp
import pennylane as qml
import numpy as np
import time

from qml_essentials.yaqsi import (
    QuantumScript,
    evolve,
    partial_trace,
    marginalize_probs,
    build_parity_observable,
)
from qml_essentials.operations import (
    H,
    CX,
    CCX,
    CY,
    CZ,
    CRX,
    CRY,
    CRZ,
    Rot,
    RX,
    RY,
    RZ,
    PauliX,
    PauliZ,
    Hermitian,
    ParametrizedHamiltonian,
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
    time_evol = evolve(Hermitian(matrix=PauliZ._matrix, wires=0))
    time_evol(t=t, wires=0)


def evol_circuit_plus(t):
    H(wires=0)  # prepare |+⟩
    time_evol = evolve(Hermitian(matrix=PauliZ._matrix, wires=0))
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
# New gate tests  (CY, CZ, CRX, CRY, CRZ, Rot) — validated against PennyLane
# ---------------------------------------------------------------------------


def _pennylane_probs(circuit_fn, n_qubits=2) -> np.ndarray:
    """Run a PennyLane circuit and return the probability vector."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_fn()
        return qml.probs(wires=list(range(n_qubits)))

    return np.array(qnode())


@pytest.mark.unittest
def test_cy_matches_pennylane() -> None:
    """CY gate probabilities match PennyLane."""

    def yaqsi_circuit():
        H(wires=0)
        CY(wires=[0, 1])

    def pl_circuit():
        qml.Hadamard(wires=0)
        qml.CY(wires=[0, 1])

    script = QuantumScript(f=yaqsi_circuit)
    probs_ours = np.array(script.execute(type="probs"))
    probs_pl = _pennylane_probs(pl_circuit)

    assert np.allclose(
        probs_ours, probs_pl, atol=1e-10
    ), f"CY mismatch:\nours = {probs_ours}\nPL   = {probs_pl}"


@pytest.mark.unittest
def test_cz_matches_pennylane() -> None:
    """CZ gate probabilities match PennyLane."""

    def yaqsi_circuit():
        H(wires=0)
        H(wires=1)
        CZ(wires=[0, 1])

    def pl_circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.CZ(wires=[0, 1])

    script = QuantumScript(f=yaqsi_circuit)
    probs_ours = np.array(script.execute(type="probs"))
    probs_pl = _pennylane_probs(pl_circuit)

    assert np.allclose(
        probs_ours, probs_pl, atol=1e-10
    ), f"CZ mismatch:\nours = {probs_ours}\nPL   = {probs_pl}"


@pytest.mark.unittest
def test_crx_matches_pennylane() -> None:
    """CRX gate probabilities match PennyLane for a non-trivial angle."""
    theta = 1.3

    def yaqsi_circuit():
        H(wires=0)
        CRX(theta, wires=[0, 1])

    def pl_circuit():
        qml.Hadamard(wires=0)
        qml.CRX(theta, wires=[0, 1])

    script = QuantumScript(f=yaqsi_circuit)
    probs_ours = np.array(script.execute(type="probs"))
    probs_pl = _pennylane_probs(pl_circuit)

    assert np.allclose(
        probs_ours, probs_pl, atol=1e-10
    ), f"CRX mismatch:\nours = {probs_ours}\nPL   = {probs_pl}"


@pytest.mark.unittest
def test_cry_matches_pennylane() -> None:
    """CRY gate probabilities match PennyLane for a non-trivial angle."""
    theta = 0.9

    def yaqsi_circuit():
        H(wires=0)
        CRY(theta, wires=[0, 1])

    def pl_circuit():
        qml.Hadamard(wires=0)
        qml.CRY(theta, wires=[0, 1])

    script = QuantumScript(f=yaqsi_circuit)
    probs_ours = np.array(script.execute(type="probs"))
    probs_pl = _pennylane_probs(pl_circuit)

    assert np.allclose(
        probs_ours, probs_pl, atol=1e-10
    ), f"CRY mismatch:\nours = {probs_ours}\nPL   = {probs_pl}"


@pytest.mark.unittest
def test_crz_matches_pennylane() -> None:
    """CRZ gate probabilities match PennyLane for a non-trivial angle."""
    theta = 2.1

    def yaqsi_circuit():
        H(wires=0)
        CRZ(theta, wires=[0, 1])

    def pl_circuit():
        qml.Hadamard(wires=0)
        qml.CRZ(theta, wires=[0, 1])

    script = QuantumScript(f=yaqsi_circuit)
    probs_ours = np.array(script.execute(type="probs"))
    probs_pl = _pennylane_probs(pl_circuit)

    assert np.allclose(
        probs_ours, probs_pl, atol=1e-10
    ), f"CRZ mismatch:\nours = {probs_ours}\nPL   = {probs_pl}"


@pytest.mark.unittest
def test_rot_matches_pennylane() -> None:
    """Rot(φ, θ, ω) = RZ(ω)·RY(θ)·RZ(φ) must match PennyLane's Rot gate."""
    phi, theta, omega = 0.4, 1.2, 2.5

    def yaqsi_circuit():
        Rot(phi, theta, omega, wires=0)

    def pl_circuit():
        qml.Rot(phi, theta, omega, wires=0)

    script = QuantumScript(f=yaqsi_circuit)
    probs_ours = np.array(script.execute(type="probs"))
    probs_pl = _pennylane_probs(pl_circuit, n_qubits=1)

    assert np.allclose(
        probs_ours, probs_pl, atol=1e-10
    ), f"Rot mismatch:\nours = {probs_ours}\nPL   = {probs_pl}"


@pytest.mark.unittest
def test_rot_decomposition_matches_individual_gates() -> None:
    """Rot(φ, θ, ω) applied as one gate equals sequential RZ·RY·RZ."""
    phi, theta, omega = 0.7, 1.5, 0.3

    def rot_circuit():
        Rot(phi, theta, omega, wires=0)

    def decomposed_circuit():
        RZ(phi, wires=0)
        RY(theta, wires=0)
        RZ(omega, wires=0)

    script_rot = QuantumScript(f=rot_circuit)
    script_dec = QuantumScript(f=decomposed_circuit)

    state_rot = np.array(script_rot.execute(type="state"))
    state_dec = np.array(script_dec.execute(type="state"))

    # Equal up to global phase
    phase = (
        state_rot[0] / state_dec[0]
        if abs(state_dec[0]) > 1e-10
        else state_rot[1] / state_dec[1]
    )
    assert np.allclose(
        state_rot, phase * state_dec, atol=1e-10
    ), f"Rot decomposition mismatch:\nrot = {state_rot}\ndec = {state_dec}"


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


# ---------------------------------------------------------------------------
# Batched execution tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_batched_expval_matches_sequential() -> None:
    """
    Batched execute (in_axes) must produce identical results to calling
    execute sequentially for each sample.

    This is the primary correctness check: jax.vmap must not introduce
    any numerical difference vs. the single-sample path.
    """
    script = QuantumScript(f=parameterized_circuit)
    thetas = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0])

    sequential = jnp.stack(
        [script.execute(type="expval", obs=[PauliZ(0)], args=(t,)) for t in thetas]
    )
    batched = script.execute(
        type="expval",
        obs=[PauliZ(0)],
        args=(thetas,),
        in_axes=(0,),
    )

    assert (
        batched.shape == sequential.shape
    ), f"Shape mismatch: batched {batched.shape} vs sequential {sequential.shape}"
    assert jnp.allclose(batched, sequential, atol=1e-6)


@pytest.mark.unittest
def test_batched_expval_values() -> None:
    """
    RX(θ)|0⟩ → ⟨Z⟩ = cos(θ) must hold element-wise across a batch.

    Mirrors the B_P (parameter-batch) axis from model.py where params
    has shape (n_layers, n_params, B_P) and in_axes=(2, ...).
    """
    script = QuantumScript(f=parameterized_circuit)
    thetas = jnp.linspace(0.0, jnp.pi, 9)

    results = script.execute(
        type="expval",
        obs=[PauliZ(0)],
        args=(thetas,),
        in_axes=(0,),
    )

    assert results.shape == (9, 1), f"Expected (9,1), got {results.shape}"
    assert jnp.allclose(results[:, 0], jnp.cos(thetas), atol=1e-6)


@pytest.mark.unittest
def test_batched_probs() -> None:
    """
    Batch over two extreme angles: RX(0)|0⟩ = |0⟩ and RX(π)|0⟩ ≈ i|1⟩.
    Probabilities must be [1,0] and [0,1] respectively.
    """
    script = QuantumScript(f=parameterized_circuit)
    thetas = jnp.array([0.0, jnp.pi])

    results = script.execute(type="probs", args=(thetas,), in_axes=(0,))

    assert results.shape == (2, 2), f"Expected (2,2), got {results.shape}"
    assert jnp.allclose(results[0], jnp.array([1.0, 0.0]), atol=1e-6)
    assert jnp.allclose(results[1], jnp.array([0.0, 1.0]), atol=1e-6)


@pytest.mark.unittest
def test_batched_gradient() -> None:
    """
    jax.grad must compose correctly with jax.vmap through execute.

    d/dθ_i mean_i(⟨Z⟩_i) = -sin(θ_i) / B, which validates that the
    gradient flows through the batched path without breaking.
    This is the core requirement for training with batched parameters.
    """
    script = QuantumScript(f=parameterized_circuit)
    thetas = jnp.array([0.3, 0.7, 1.2])

    def loss(thetas):
        results = script.execute(
            type="expval",
            obs=[PauliZ(0)],
            args=(thetas,),
            in_axes=(0,),
        )
        return jnp.mean(results)

    grad = jax.grad(loss)(thetas)
    expected = -jnp.sin(thetas) / len(thetas)
    assert jnp.allclose(grad, expected, atol=1e-6)


@pytest.mark.unittest
def test_batched_broadcast_none_axis() -> None:
    """
    in_axes=None for an argument means it is broadcast across the batch
    (not sliced), matching jax.vmap convention.

    We pass a fixed observable circuit with no args and a separately
    batched scalar to confirm None is handled without error.
    """

    def two_arg_circuit(theta, phi):
        RX(theta, wires=0)
        RX(phi, wires=0)

    script = QuantumScript(f=two_arg_circuit)
    thetas = jnp.linspace(0.0, 1.0, 5)  # batched — axis 0
    phi = jnp.array(0.5)  # broadcast — None

    results = script.execute(
        type="expval",
        obs=[PauliZ(0)],
        args=(thetas, phi),
        in_axes=(0, None),
    )

    # Each result should equal cos(theta + phi) = cos(theta + 0.5)
    expected = jnp.array(
        [
            script.execute(type="expval", obs=[PauliZ(0)], args=(t, phi))[0]
            for t in thetas
        ]
    )
    assert results.shape == (5, 1), f"Expected (5,1), got {results.shape}"
    assert jnp.allclose(results[:, 0], expected, atol=1e-6)


@pytest.mark.unittest
def test_batched_in_axes_mismatch_raises() -> None:
    """in_axes length != args length must raise a clear ValueError."""
    script = QuantumScript(f=parameterized_circuit)
    with pytest.raises(ValueError, match="in_axes has"):
        script.execute(
            type="expval",
            obs=[PauliZ(0)],
            args=(jnp.array([0.5]),),
            in_axes=(0, None),  # 2 entries but only 1 arg
        )


@pytest.mark.unittest
def test_batched_multi_qubit() -> None:
    """
    Batching works on multi-qubit circuits (GHZ family).

    We batch a rotation angle applied before the Bell circuit and verify
    the output probabilities change as expected.  When theta=0 we get the
    standard Bell state; when theta=pi/2 the H·RX(π/2) combination
    produces a different distribution — we just check the batch dimension
    and that probabilities sum to 1.
    """

    def rotated_bell(theta):
        RX(theta, wires=0)
        H(wires=0)
        CX(wires=[0, 1])

    script = QuantumScript(f=rotated_bell)
    thetas = jnp.linspace(0.0, jnp.pi, 4)

    results = script.execute(type="probs", args=(thetas,), in_axes=(0,))

    assert results.shape == (4, 4), f"Expected (4,4), got {results.shape}"
    # Each probability vector must sum to 1
    assert jnp.allclose(results.sum(axis=1), jnp.ones(4), atol=1e-8)


# ---------------------------------------------------------------------------
# partial_trace tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_partial_trace_bell_keep_0() -> None:
    """Tracing out qubit 1 of the Bell state gives the maximally mixed state."""
    script = QuantumScript(f=bell_circuit)
    rho = script.execute(type="density")
    rho_0 = partial_trace(rho, n_qubits=2, keep=[0])
    expected = 0.5 * jnp.eye(2, dtype=jnp.complex128)
    assert jnp.allclose(rho_0, expected, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_bell_keep_1() -> None:
    """Tracing out qubit 0 of the Bell state gives the maximally mixed state."""
    script = QuantumScript(f=bell_circuit)
    rho = script.execute(type="density")
    rho_1 = partial_trace(rho, n_qubits=2, keep=[1])
    expected = 0.5 * jnp.eye(2, dtype=jnp.complex128)
    assert jnp.allclose(rho_1, expected, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_product_state() -> None:
    """Tracing a product state |0⟩|+⟩ over qubit 0 yields |+⟩⟨+|."""

    def circuit(*a, **kw):
        H(wires=1)  # qubit 1 -> |+⟩, qubit 0 stays |0⟩

    script = QuantumScript(f=circuit)
    rho = script.execute(type="density")
    rho_1 = partial_trace(rho, n_qubits=2, keep=[1])
    # |+⟩⟨+| = 0.5 * [[1, 1], [1, 1]]
    expected = 0.5 * jnp.ones((2, 2), dtype=jnp.complex128)
    assert jnp.allclose(rho_1, expected, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_keep_all() -> None:
    """Keeping all qubits returns the original density matrix."""
    script = QuantumScript(f=bell_circuit)
    rho = script.execute(type="density")
    rho_all = partial_trace(rho, n_qubits=2, keep=[0, 1])
    assert jnp.allclose(rho_all, rho, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_batched() -> None:
    """partial_trace handles a batched (B, d, d) density matrix."""

    def rx_circuit(theta):
        RX(theta, wires=0)

    script = QuantumScript(f=rx_circuit)
    thetas = jnp.array([0.0, jnp.pi / 2, jnp.pi])

    rho_batch = jnp.stack([script.execute(type="density", args=(t,)) for t in thetas])
    assert rho_batch.shape == (3, 2, 2)

    # For a 1-qubit system, keeping qubit 0 = identity operation
    rho_traced = partial_trace(rho_batch, n_qubits=1, keep=[0])
    assert jnp.allclose(rho_traced, rho_batch, atol=1e-10)


# ---------------------------------------------------------------------------
# marginalize_probs tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_marginalize_probs_bell_keep_0() -> None:
    """Marginalizing qubit 1 of the Bell state gives [0.5, 0.5]."""
    script = QuantumScript(f=bell_circuit)
    probs = script.execute(type="probs")
    marginal = marginalize_probs(probs, n_qubits=2, keep=[0])
    assert jnp.allclose(marginal, jnp.array([0.5, 0.5]), atol=1e-10)


@pytest.mark.unittest
def test_marginalize_probs_keep_all() -> None:
    """Keeping all qubits returns the original probabilities."""
    script = QuantumScript(f=bell_circuit)
    probs = script.execute(type="probs")
    full = marginalize_probs(probs, n_qubits=2, keep=[0, 1])
    assert jnp.allclose(full, probs, atol=1e-10)


@pytest.mark.unittest
def test_marginalize_probs_batched() -> None:
    """marginalize_probs handles batched (B, 2**n) probability vectors."""

    def rx_circuit(theta):
        RX(theta, wires=0)
        H(wires=1)

    script = QuantumScript(f=rx_circuit)
    thetas = jnp.array([0.0, jnp.pi / 2])

    probs_batch = jnp.stack([script.execute(type="probs", args=(t,)) for t in thetas])
    assert probs_batch.shape == (2, 4)

    marginal = marginalize_probs(probs_batch, n_qubits=2, keep=[0])
    assert marginal.shape == (2, 2)
    # Each row must sum to 1
    assert jnp.allclose(marginal.sum(axis=1), jnp.ones(2), atol=1e-10)


# ---------------------------------------------------------------------------
# build_parity_observable tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_parity_observable_single_qubit() -> None:
    """Single-qubit parity observable is just Z."""
    obs = build_parity_observable([0])
    assert obs.matrix.shape == (2, 2)
    assert jnp.allclose(obs.matrix, PauliZ._matrix, atol=1e-10)


@pytest.mark.unittest
def test_parity_observable_two_qubit() -> None:
    """Two-qubit parity observable is Z⊗Z."""
    obs = build_parity_observable([0, 1])
    expected = jnp.kron(PauliZ._matrix, PauliZ._matrix)
    assert obs.matrix.shape == (4, 4)
    assert jnp.allclose(obs.matrix, expected, atol=1e-10)


@pytest.mark.unittest
def test_parity_observable_not_on_tape() -> None:
    """build_parity_observable must not add operations to the tape."""
    from qml_essentials.tape import recording

    with recording() as tape:
        _ = build_parity_observable([0, 1])
    assert len(tape) == 0, "Parity observable should not record on the tape"


# ---------------------------------------------------------------------------
# Hermitian record=False tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_hermitian_record_false_not_on_tape() -> None:
    """Hermitian(record=False) must not appear on the tape."""
    from qml_essentials.tape import recording

    with recording() as tape:
        _ = Hermitian(matrix=PauliZ._matrix, wires=0, record=False)
    assert len(tape) == 0


@pytest.mark.unittest
def test_hermitian_record_true_on_tape() -> None:
    """Hermitian(record=True) (default) does appear on the tape."""
    from qml_essentials.tape import recording

    with recording() as tape:
        _ = Hermitian(matrix=PauliZ._matrix, wires=0)
    assert len(tape) == 1


# ---------------------------------------------------------------------------
# ParametrizedHamiltonian tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_parametrized_hamiltonian_creation() -> None:
    """coeff_fn * Hermitian produces a ParametrizedHamiltonian."""

    def coeff(p, t):
        return p * t

    herm = Hermitian(matrix=PauliZ._matrix, wires=0, record=False)
    ph = coeff * herm

    assert isinstance(ph, ParametrizedHamiltonian)
    assert ph.coeff_fn is coeff
    assert jnp.allclose(ph.H_mat, PauliZ._matrix)
    assert ph.wires == [0]


@pytest.mark.unittest
def test_parametrized_hamiltonian_non_callable_raises() -> None:
    """Multiplying a non-callable with Hermitian raises TypeError."""
    herm = Hermitian(matrix=PauliZ._matrix, wires=0, record=False)
    with pytest.raises(TypeError, match="callable"):
        _ = 3.14 * herm


# ---------------------------------------------------------------------------
# Time-dependent evolve tests
# ---------------------------------------------------------------------------


@pytest.mark.unittest
def test_evolve_parametrized_constant_matches_static() -> None:
    """
    A constant coefficient f(p, t) = 1 reduces to the static case:
    U = exp(-i T H).  The ODE result must match the matrix expm result.
    """
    Z = PauliZ._matrix
    T_val = 0.7

    # Static evolve
    def static_circuit(t):
        gate = evolve(Hermitian(matrix=Z, wires=0, record=False))
        gate(t=t, wires=0)

    script_s = QuantumScript(f=static_circuit)
    state_static = script_s.execute(type="state", args=(T_val,))

    # Parametrized evolve with f(p, t) = 1
    def const_coeff(p, t):
        return 1.0

    def param_circuit(T):
        ph = const_coeff * Hermitian(matrix=Z, wires=0, record=False)
        evolve(ph)([0.0], T)  # coeff_args=[0.0] unused

    script_p = QuantumScript(f=param_circuit)
    state_param = script_p.execute(type="state", args=(T_val,))

    assert jnp.allclose(
        state_static, state_param, atol=1e-6
    ), f"Static: {state_static}, Param: {state_param}"


@pytest.mark.unittest
def test_evolve_parametrized_unitarity() -> None:
    """The unitary from time-dependent evolve satisfies U†U = I."""

    def coeff(p, t):
        return p[0] * jnp.cos(p[1] * t)

    Z = PauliZ._matrix
    ph = coeff * Hermitian(matrix=Z, wires=0, record=False)

    gate_factory = evolve(ph)
    op = gate_factory([jnp.array([2.0, 3.0])], 1.5)
    U = op.matrix
    assert jnp.allclose(U.conj().T @ U, jnp.eye(2), atol=1e-8), "U is not unitary"


@pytest.mark.unittest
def test_evolve_parametrized_differentiable() -> None:
    """
    jax.grad can differentiate through the ODE-based evolve.
    For H = Z and f(p, t) = p, the unitary is exp(-i p T Z) giving
    ⟨Z⟩ = 1 (independent of p for |0⟩), so d⟨Z⟩/dp = 0.
    For ⟨X⟩ = 0 starting from |0⟩ (Z eigenstate), also zero.
    Test with |+⟩ state for non-trivial gradient.
    """

    def coeff(p, t):
        return p

    Z = PauliZ._matrix

    def cost(p):
        def circuit(p_val):
            H(wires=0)  # prepare |+⟩
            ph = coeff * Hermitian(matrix=Z, wires=0, record=False)
            evolve(ph)([p_val], 1.0)

        script = QuantumScript(f=circuit)
        return script.execute(type="expval", obs=[PauliX(0)], args=(p,))[0]

    p_val = jnp.array(0.5)
    # ⟨X⟩ = cos(2p) for H exp(-ipZ)|0⟩, so d⟨X⟩/dp = -2 sin(2p)
    grad = jax.grad(cost)(p_val)
    expected_grad = -2.0 * jnp.sin(2.0 * p_val)
    assert jnp.allclose(
        grad, expected_grad, atol=1e-4
    ), f"Grad: {grad}, expected: {expected_grad}"


@pytest.mark.unittest
def test_evolve_parametrized_on_tape() -> None:
    """The EvolvedOp from time-dependent evolve is recorded on the tape."""
    from qml_essentials.tape import recording

    def coeff(p, t):
        return 1.0

    ph = coeff * Hermitian(matrix=PauliZ._matrix, wires=0, record=False)

    with recording() as tape:
        evolve(ph)([0.0], 1.0)

    assert len(tape) == 1, f"Expected 1 op on tape, got {len(tape)}"
    assert tape[0].wires == [0]


@pytest.mark.unittest
def test_evolve_parametrized_hermitian_not_on_tape() -> None:
    """
    When building a ParametrizedHamiltonian inside a recording context,
    the Hermitian used for construction must NOT appear on the tape —
    only the EvolvedOp should.
    """
    from qml_essentials.tape import recording

    def coeff(p, t):
        return 1.0

    with recording() as tape:
        herm = Hermitian(matrix=PauliZ._matrix, wires=0, record=False)
        ph = coeff * herm
        evolve(ph)([0.0], 1.0)

    # Only the EvolvedOp should be on the tape
    assert len(tape) == 1, (
        f"Expected 1 op (EvolvedOp), got {len(tape)}: "
        f"{[type(o).__name__ for o in tape]}"
    )


@pytest.mark.unittest
def test_evolve_type_error() -> None:
    """evolve() with an unsupported type raises TypeError."""
    with pytest.raises(TypeError, match="evolve"):
        evolve("not a hamiltonian")


@pytest.mark.expensive
@pytest.mark.unittest
@pytest.mark.parametrize("mode", ["probs", "expval", "state", "density"])
def test_benchmark(mode) -> None:
    """Benchmark comparison with pennylane framework (non-parametric circuit)"""

    n_qubits = 6

    # Yaqsi
    def yaqsi_circuit():
        for i in range(n_qubits):
            H(wires=i)
        for i in range(n_qubits):
            CX(wires=[i, (i + 1) % n_qubits])

    obs = [PauliZ(wires=i, record=False) for i in range(n_qubits)]

    # Warmup
    _ = QuantumScript(f=yaqsi_circuit).execute(type=mode, obs=obs)

    start = time.time()
    for _ in range(100):
        res_ys = QuantumScript(f=yaqsi_circuit).execute(type=mode, obs=obs)
    t_ys = (time.time() - start) / 100
    print(f"Yaqsi {mode} ({n_qubits}q, avg 100): {t_ys*1000:.2f} ms")

    # PennyLane — use a separate QNode per mode since the return differs
    dev = qml.device("default.qubit", wires=n_qubits)

    pl_return_map = {
        "density": lambda: qml.density_matrix(wires=range(n_qubits)),
        "state": lambda: qml.state(),
        "probs": lambda: qml.probs(wires=range(n_qubits)),
        "expval": lambda: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)],
    }

    @qml.qnode(dev)
    def pl_circuit():
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return pl_return_map[mode]()

    # Warmup
    _ = pl_circuit()

    start = time.time()
    for _ in range(100):
        res_pl = pl_circuit()
    t_pl = (time.time() - start) / 100
    print(f"PennyLane {mode} ({n_qubits}q, avg 100): {t_pl*1000:.2f} ms")
    print(f"Ratio yaqsi/pl: {t_ys/t_pl:.1f}x")
    print(f"Results match: {jnp.allclose(res_ys, jnp.array(res_pl), atol=1e-10)}")


@pytest.mark.expensive
@pytest.mark.unittest
@pytest.mark.parametrize("mode", ["probs", "expval", "state", "density"])
def test_batch_benchmark(mode) -> None:
    """Benchmark comparison with pennylane framework (parametric, batched).

    Simulates a realistic training loop: parameters change every iteration,
    batch of 100 samples is vmapped, and the same QuantumScript instance is
    reused across iterations to test JIT compilation caching.
    """

    n_qubits = 6
    n_iters = 100
    batch_size = 100

    # Pre-generate different parameters for each iteration to simulate
    # a training loop where params change every step.
    rng = jax.random.PRNGKey(42)
    all_phis = jax.random.uniform(
        rng, shape=(n_iters, batch_size), minval=-jnp.pi, maxval=jnp.pi
    )

    obs = [PauliZ(wires=i, record=False) for i in range(n_qubits)]

    # --- Yaqsi ---
    def yaqsi_circuit(phi):
        for i in range(n_qubits):
            H(wires=i)
        for i in range(n_qubits):
            CRX(phi, wires=[i, (i + 1) % n_qubits])

    # Reuse the same QuantumScript to benefit from JIT compilation caching
    script = QuantumScript(f=yaqsi_circuit)

    # Warmup (triggers first compilation)
    _ = script.execute(type=mode, obs=obs, args=(all_phis[0],), in_axes=(0,))

    start = time.time()
    for i in range(n_iters):
        res_ys = script.execute(type=mode, obs=obs, args=(all_phis[i],), in_axes=(0,))
    t_ys = (time.time() - start) / n_iters
    print(
        f"Yaqsi {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_ys*1000:.2f} ms"
    )

    # --- PennyLane ---
    dev = qml.device("default.qubit", wires=n_qubits)

    pl_return_map = {
        "density": lambda: qml.density_matrix(wires=range(n_qubits)),
        "state": lambda: qml.state(),
        "probs": lambda: qml.probs(wires=range(n_qubits)),
        "expval": lambda: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)],
    }

    @qml.qnode(dev)
    def pl_circuit(phi):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.CRX(phi, wires=[i, (i + 1) % n_qubits])
        return pl_return_map[mode]()

    # Warmup
    _ = pl_circuit(all_phis[0])

    start = time.time()
    for i in range(n_iters):
        res_pl = pl_circuit(all_phis[i])
    t_pl = (time.time() - start) / n_iters
    print(
        f"PennyLane {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_pl*1000:.2f} ms"
    )
    print(f"Ratio yaqsi/pl: {t_ys/t_pl:.1f}x")
    res_pl_arr = jnp.array(res_pl)
    # PennyLane expval returns (n_obs, batch) while yaqsi returns (batch, n_obs)
    if res_pl_arr.shape != res_ys.shape:
        res_pl_arr = res_pl_arr.T
    print(f"Results match: {jnp.allclose(res_ys, res_pl_arr, atol=1e-10)}")
