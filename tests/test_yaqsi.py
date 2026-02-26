import pytest
import jax

import jax.numpy as jnp
import pennylane as qml
import numpy as np
import time


from qml_essentials.yaqsi import (
    Script,
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
from qml_essentials.math import fidelity, trace_distance, phase_difference

import logging

logger = logging.getLogger(__name__)

jax.config.update("jax_enable_x64", True)  # tests use atol=1e-10


def bell_circuit(*args, **kwargs):
    H(wires=0)
    CX(wires=[0, 1])


def param_bell_circuit(theta):
    """Simple 2-qubit circuit: H(0) CX(0,1) RZ(theta, 0)."""
    H(wires=0)
    CX(wires=[0, 1])
    RZ(theta, wires=0)


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
    Then CCX(0,1->2) doesn't flip anything useful yet, so we first create
    |11⟩ on qubits 0,1 with probability 0.5, then Toffoli flips qubit 2.

    Circuit: H(0) -> CX(0,1) -> CCX(0,1,2)
    State after H+CX:  (|00⟩+|11⟩)/√2 ⊗ |0⟩
    After CCX (flips q2 only when q0=q1=1):  (|000⟩+|111⟩)/√2
    """
    H(wires=0)
    CX(wires=[0, 1])
    CCX(wires=[0, 1, 2])


def parametrized_circuit(theta):
    RX(theta, wires=0)


def evol_circuit(t):
    time_evol = evolve(Hermitian(matrix=PauliZ._matrix, wires=0))
    time_evol(t=t, wires=0)


def evol_circuit_plus(t):
    H(wires=0)  # prepare |+⟩
    time_evol = evolve(Hermitian(matrix=PauliZ._matrix, wires=0))
    time_evol(t=t, wires=0)


def _pennylane_probs(circuit_fn, n_qubits=2) -> np.ndarray:
    """Run a PennyLane circuit and return the probability vector."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_fn()
        return qml.probs(wires=list(range(n_qubits)))

    return np.array(qnode())


# Helper: run a 1-qubit circuit through PennyLane default.mixed and return ρ
def _pennylane_density(circuit_fn, n_qubits=1) -> np.ndarray:
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def qnode():
        circuit_fn()
        return qml.density_matrix(wires=list(range(n_qubits)))

    return np.array(qnode())


@pytest.mark.unittest
def test_expval_bell_x() -> None:
    """Bell state (|00⟩+|11⟩)/√2 has ⟨Xᵢ⟩ = 0 (no single-qubit X bias)."""
    script = Script(f=bell_circuit)
    res = script.execute(type="expval", obs=[PauliX(0), PauliX(1)])
    assert jnp.allclose(res, jnp.array([0.0, 0.0]), atol=1e-10)


@pytest.mark.unittest
def test_probs_bell() -> None:
    """Bell state yields |00⟩ and |11⟩ each with probability 0.5."""
    script = Script(f=bell_circuit)
    probs = script.execute(type="probs")
    assert jnp.allclose(probs, jnp.array([0.5, 0.0, 0.0, 0.5]), atol=1e-10)


@pytest.mark.unittest
def test_expval_bell_z() -> None:
    """Bell state has ⟨Zᵢ⟩ = 0 for each qubit individually."""
    script = Script(f=bell_circuit)
    res_zz = script.execute(type="expval", obs=[PauliZ(0), PauliZ(1)])
    assert jnp.allclose(res_zz, jnp.array([0.0, 0.0]), atol=1e-10)


@pytest.mark.unittest
def test_parametrized_expval() -> None:
    """RX(θ)|0⟩ gives ⟨Z⟩ = cos(θ)."""
    script = Script(f=parametrized_circuit)
    theta_val = jnp.array(0.5)

    def cost(theta):
        return script.execute(type="expval", obs=[PauliZ(0)], args=(theta,))[0]

    assert jnp.allclose(cost(theta_val), jnp.cos(theta_val), atol=1e-6)


@pytest.mark.unittest
def test_jax_gradient() -> None:
    """Gradient of ⟨Z⟩ w.r.t. θ for RX(θ)|0⟩ equals -sin(θ)."""
    script = Script(f=parametrized_circuit)
    theta_val = jnp.array(0.5)

    def cost(theta):
        return script.execute(type="expval", obs=[PauliZ(0)], args=(theta,))[0]

    grad_fn = jax.grad(cost)
    expected_grad = -jnp.sin(theta_val)
    assert jnp.allclose(grad_fn(theta_val), expected_grad, atol=1e-6)


@pytest.mark.unittest
def test_evolve_from_zero() -> None:
    """exp(-i·t·Z)|0⟩ leaves ⟨X⟩ = 0 since |0⟩ is a Z eigenstate."""
    script = Script(f=evol_circuit)
    res = script.execute(type="expval", obs=[PauliX(0)], args=(0.3,))
    assert jnp.allclose(res, jnp.array([0.0]), atol=1e-6)


@pytest.mark.unittest
def test_evolve_from_plus() -> None:
    """exp(-i·t·Z)|+⟩ gives |⟨X⟩| = cos(2t)."""
    t = 0.3
    script = Script(f=evol_circuit_plus)
    res = script.execute(type="expval", obs=[PauliX(0)], args=(t,))
    assert jnp.allclose(jnp.abs(res), jnp.cos(2 * t), atol=1e-6)


@pytest.mark.unittest
def test_probs_ghz_3() -> None:
    """3-qubit GHZ state has prob 0.5 on |000⟩ and |111⟩, zero elsewhere."""
    script = Script(f=ghz_circuit_3)
    probs = script.execute(type="probs")
    expected = jnp.zeros(8).at[0].set(0.5).at[7].set(0.5)
    assert jnp.allclose(probs, expected, atol=1e-10)


@pytest.mark.unittest
def test_expval_ghz_3_z() -> None:
    """Each qubit of a 3-qubit GHZ state has ⟨Zᵢ⟩ = 0."""
    script = Script(f=ghz_circuit_3)
    obs = [PauliZ(0), PauliZ(1), PauliZ(2)]
    res = script.execute(type="expval", obs=obs)
    assert jnp.allclose(res, jnp.zeros(3), atol=1e-10)


@pytest.mark.unittest
def test_probs_ghz_4() -> None:
    """4-qubit GHZ state has prob 0.5 on |0000⟩ and |1111⟩, zero elsewhere."""
    script = Script(f=ghz_circuit_4)
    probs = script.execute(type="probs")
    expected = jnp.zeros(16).at[0].set(0.5).at[15].set(0.5)
    assert jnp.allclose(probs, expected, atol=1e-10)


@pytest.mark.unittest
def test_probs_ghz_toffoli() -> None:
    """3-qubit GHZ via Toffoli matches the CNOT-chain result."""
    script_cnot = Script(f=ghz_circuit_3)
    script_toff = Script(f=ghz_toffoli_3)
    probs_cnot = script_cnot.execute(type="probs")
    probs_toff = script_toff.execute(type="probs")
    assert jnp.allclose(probs_cnot, probs_toff, atol=1e-10)


@pytest.mark.unittest
def test_state_ghz_3_non_adjacent_wires() -> None:
    """CX on non-adjacent wires [0,2] still produces the correct entangled state."""

    def circuit(*args, **kwargs):
        H(wires=0)
        CX(wires=[0, 2])  # control=0, target=2 (skip qubit 1)

    script = Script(f=circuit)
    probs = script.execute(type="probs")
    # |000⟩ (index 0) and |101⟩ (index 5 = 0b101) each with prob 0.5
    expected = jnp.zeros(8).at[0].set(0.5).at[5].set(0.5)
    assert jnp.allclose(probs, expected, atol=1e-10)


@pytest.mark.unittest
def test_density_pure_state_is_projector() -> None:
    """
    A density matrix from a pure circuit satisfies ρ² = ρ (idempotent)
    and Tr(ρ) = 1.
    """
    script = Script(f=bell_circuit)
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
    script = Script(f=bell_circuit)
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
    script = Script(f=ghz_circuit_3)
    probs = script.execute(type="probs")
    rho = script.execute(type="density")

    assert jnp.allclose(jnp.real(jnp.diag(rho)), probs, atol=1e-10)


@pytest.mark.unittest
def test_density_is_hermitian() -> None:
    """Density matrix must satisfy ρ = ρ†."""
    script = Script(f=ghz_circuit_4)
    rho = script.execute(type="density")

    assert jnp.allclose(rho, jnp.conj(rho.T), atol=1e-10)


@pytest.mark.unittest
def test_density_expval_matches_statevector() -> None:
    """
    Tr(O ρ) via the density path must equal ⟨ψ|O|ψ⟩ from the statevector path
    for any observable and any pure circuit.
    """
    script = Script(f=parametrized_circuit)
    theta_val = jnp.array(0.7)

    ev_pure = script.execute(type="expval", obs=[PauliZ(0)], args=(theta_val,))[0]

    rho = script.execute(type="density", args=(theta_val,))
    # Tr(Z ρ) — build Z in the 1-qubit Hilbert space directly
    Z_mat = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    ev_density = jnp.real(jnp.trace(Z_mat @ rho))

    assert jnp.allclose(ev_pure, ev_density, atol=1e-8)


@pytest.mark.unittest
def test_cy_matches_pennylane() -> None:
    """CY gate probabilities match PennyLane."""

    def yaqsi_circuit():
        H(wires=0)
        CY(wires=[0, 1])

    def pl_circuit():
        qml.Hadamard(wires=0)
        qml.CY(wires=[0, 1])

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script_rot = Script(f=rot_circuit)
    script_dec = Script(f=decomposed_circuit)

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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=yaqsi_circuit)
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

    script = Script(f=noisy_circuit)
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

    script = Script(f=noisy_bell)
    rho = script.execute(type="density")

    assert rho.shape == (4, 4)
    assert jnp.allclose(jnp.trace(rho), 1.0, atol=1e-10), "Tr(ρ) ≠ 1"
    assert jnp.allclose(rho, jnp.conj(rho.T), atol=1e-10), "ρ not Hermitian"
    # Purity: Tr(ρ²) ≤ 1 for a mixed state, < 1 when noise is present
    purity = jnp.real(jnp.trace(rho @ rho))
    assert purity <= 1.0 + 1e-10, f"purity {purity} > 1"
    assert purity < 1.0 - 1e-6, f"purity {purity} ≈ 1: channel had no effect?"


@pytest.mark.unittest
def test_batched_expval_matches_sequential() -> None:
    """
    Batched execute (in_axes) must produce identical results to calling
    execute sequentially for each sample.

    This is the primary correctness check: jax.vmap must not introduce
    any numerical difference vs. the single-sample path.
    """
    script = Script(f=parametrized_circuit)
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
    RX(θ)|0⟩ -> ⟨Z⟩ = cos(θ) must hold element-wise across a batch.

    Mirrors the B_P (parameter-batch) axis from model.py where params
    has shape (n_layers, n_params, B_P) and in_axes=(2, ...).
    """
    script = Script(f=parametrized_circuit)
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
    script = Script(f=parametrized_circuit)
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
    script = Script(f=parametrized_circuit)
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

    script = Script(f=two_arg_circuit)
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
    script = Script(f=parametrized_circuit)
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

    script = Script(f=rotated_bell)
    thetas = jnp.linspace(0.0, jnp.pi, 4)

    results = script.execute(type="probs", args=(thetas,), in_axes=(0,))

    assert results.shape == (4, 4), f"Expected (4,4), got {results.shape}"
    # Each probability vector must sum to 1
    assert jnp.allclose(results.sum(axis=1), jnp.ones(4), atol=1e-8)


@pytest.mark.unittest
def test_partial_trace_bell_keep_0() -> None:
    """Tracing out qubit 1 of the Bell state gives the maximally mixed state."""
    script = Script(f=bell_circuit)
    rho = script.execute(type="density")
    rho_0 = partial_trace(rho, n_qubits=2, keep=[0])
    expected = 0.5 * jnp.eye(2, dtype=jnp.complex128)
    assert jnp.allclose(rho_0, expected, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_bell_keep_1() -> None:
    """Tracing out qubit 0 of the Bell state gives the maximally mixed state."""
    script = Script(f=bell_circuit)
    rho = script.execute(type="density")
    rho_1 = partial_trace(rho, n_qubits=2, keep=[1])
    expected = 0.5 * jnp.eye(2, dtype=jnp.complex128)
    assert jnp.allclose(rho_1, expected, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_product_state() -> None:
    """Tracing a product state |0⟩|+⟩ over qubit 0 yields |+⟩⟨+|."""

    def circuit(*a, **kw):
        H(wires=1)  # qubit 1 -> |+⟩, qubit 0 stays |0⟩

    script = Script(f=circuit)
    rho = script.execute(type="density")
    rho_1 = partial_trace(rho, n_qubits=2, keep=[1])
    # |+⟩⟨+| = 0.5 * [[1, 1], [1, 1]]
    expected = 0.5 * jnp.ones((2, 2), dtype=jnp.complex128)
    assert jnp.allclose(rho_1, expected, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_keep_all() -> None:
    """Keeping all qubits returns the original density matrix."""
    script = Script(f=bell_circuit)
    rho = script.execute(type="density")
    rho_all = partial_trace(rho, n_qubits=2, keep=[0, 1])
    assert jnp.allclose(rho_all, rho, atol=1e-10)


@pytest.mark.unittest
def test_partial_trace_batched() -> None:
    """partial_trace handles a batched (B, d, d) density matrix."""

    def rx_circuit(theta):
        RX(theta, wires=0)

    script = Script(f=rx_circuit)
    thetas = jnp.array([0.0, jnp.pi / 2, jnp.pi])

    rho_batch = jnp.stack([script.execute(type="density", args=(t,)) for t in thetas])
    assert rho_batch.shape == (3, 2, 2)

    # For a 1-qubit system, keeping qubit 0 = identity operation
    rho_traced = partial_trace(rho_batch, n_qubits=1, keep=[0])
    assert jnp.allclose(rho_traced, rho_batch, atol=1e-10)


@pytest.mark.unittest
def test_marginalize_probs_bell_keep_0() -> None:
    """Marginalizing qubit 1 of the Bell state gives [0.5, 0.5]."""
    script = Script(f=bell_circuit)
    probs = script.execute(type="probs")
    marginal = marginalize_probs(probs, n_qubits=2, keep=[0])
    assert jnp.allclose(marginal, jnp.array([0.5, 0.5]), atol=1e-10)


@pytest.mark.unittest
def test_marginalize_probs_keep_all() -> None:
    """Keeping all qubits returns the original probabilities."""
    script = Script(f=bell_circuit)
    probs = script.execute(type="probs")
    full = marginalize_probs(probs, n_qubits=2, keep=[0, 1])
    assert jnp.allclose(full, probs, atol=1e-10)


@pytest.mark.unittest
def test_marginalize_probs_batched() -> None:
    """marginalize_probs handles batched (B, 2**n) probability vectors."""

    def rx_circuit(theta):
        RX(theta, wires=0)
        H(wires=1)

    script = Script(f=rx_circuit)
    thetas = jnp.array([0.0, jnp.pi / 2])

    probs_batch = jnp.stack([script.execute(type="probs", args=(t,)) for t in thetas])
    assert probs_batch.shape == (2, 4)

    marginal = marginalize_probs(probs_batch, n_qubits=2, keep=[0])
    assert marginal.shape == (2, 2)
    # Each row must sum to 1
    assert jnp.allclose(marginal.sum(axis=1), jnp.ones(2), atol=1e-10)


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

    script_s = Script(f=static_circuit)
    state_static = script_s.execute(type="state", args=(T_val,))

    # Parametrized evolve with f(p, t) = 1
    def const_coeff(p, t):
        return 1.0

    def param_circuit(T):
        ph = const_coeff * Hermitian(matrix=Z, wires=0, record=False)
        evolve(ph)([0.0], T)  # coeff_args=[0.0] unused

    script_p = Script(f=param_circuit)
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

        script = Script(f=circuit)
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


@pytest.mark.unittest
@pytest.mark.limit_memory("1 GB")
def test_memory() -> None:
    """
    Note, this test requires memray to be activated. Run with
    pytest tests/test_yaqsi.py::test_memory -x -s --memray
    """
    n_qubits = 12

    # Yaqsi
    def yaqsi_circuit():
        for i in range(n_qubits):
            H(wires=i)
        for i in range(n_qubits):
            CX(wires=[i, (i + 1) % n_qubits])

    for _ in range(100):
        _ = Script(f=yaqsi_circuit).execute(type="density")


@pytest.mark.benchmark
@pytest.mark.unittest
@pytest.mark.parametrize(
    "mode,speedup", [("probs", 90), ("expval", 90), ("state", 70), ("density", 65)]
)
def test_mode_performances(benchmark, mode, speedup) -> None:
    """
    Note, this test requires codspeed to be activated. Run with
    pytest tests/test_yaqsi.py::test_mode_performances -x -s --codspeed
    """

    n_qubits = 6
    n_iters = 100
    batch_size = 10
    rng = jax.random.PRNGKey(1000)
    rng, subkey = jax.random.split(rng)

    # Pre-generate different parameters for each iteration to simulate
    # a training loop where params change every step.
    all_phis = jax.random.uniform(
        subkey, shape=(n_iters, batch_size), minval=-jnp.pi, maxval=jnp.pi
    )

    # --- Yaqsi ---
    def yaqsi_circuit(phi):
        for i in range(n_qubits):
            H(wires=i)
        for i in range(n_qubits):
            CRX(phi, wires=[i, (i + 1) % n_qubits])

    script = Script(f=yaqsi_circuit)

    _ = script.execute(
        type=mode,
        obs=[PauliZ(wires=i, record=False) for i in range(n_qubits)],
        args=(all_phis[0],),
        in_axes=(0,),
    )

    def ys_benchmark():
        ys_times = []
        for i in range(n_iters):
            t0 = time.perf_counter()
            res_ys = script.execute(
                type=mode,
                obs=[PauliZ(wires=i, record=False) for i in range(n_qubits)],
                args=(all_phis[i],),
                in_axes=(0,),
            )
            ys_times.append(time.perf_counter() - t0)
        return ys_times, res_ys

    ys_times, res_ys = benchmark(ys_benchmark)
    t_ys = float(np.mean(ys_times))
    std_ys = float(np.std(ys_times))

    logger.info(
        f"Yaqsi {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_ys*1000:.2f} ± {std_ys*1000:.2f} ms"
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

    _ = pl_circuit(all_phis[0])

    def pl_benchmark():
        pl_times = []
        for i in range(n_iters):
            t0 = time.perf_counter()
            res_pl = pl_circuit(all_phis[i])
            pl_times.append(time.perf_counter() - t0)
        t_pl = float(np.mean(pl_times))
        std_pl = float(np.std(pl_times))
        return t_pl, std_pl, res_pl

    t_pl, std_pl, res_pl = pl_benchmark()

    logger.info(
        f"PennyLane {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_pl*1000:.2f} ± {std_pl*1000:.2f} ms"
    )
    ratio = t_pl / t_ys
    logger.info(f"Ratio pl/yaqsi: {ratio:.2f}x")
    assert (
        ratio >= speedup
    ), f"Yaqsi not significantly faster than PennyLane. {ratio:2f}x"

    res_pl_arr = jnp.array(res_pl)
    if res_pl_arr.shape != res_ys.shape:
        res_pl_arr = res_pl_arr.T

    assert jnp.allclose(res_ys, res_pl_arr, atol=1e-10), "Results do not match"
    logger.info("Results match")


@pytest.mark.unittest
def test_shots_probs_single():
    """Shot-sampled probs should sum to 1 and have correct shape."""
    script = Script(param_bell_circuit, n_qubits=2)
    key = jax.random.PRNGKey(42)
    result = script.execute(
        type="probs",
        args=(0.5,),
        shots=4096,
        key=key,
    )
    assert result.shape == (4,), f"Expected shape (4,), got {result.shape}"
    assert jnp.allclose(result.sum(), 1.0, atol=1e-10), "Probs don't sum to 1"
    # All probabilities should be non-negative
    assert jnp.all(result >= 0), "Negative probability found"


@pytest.mark.unittest
def test_shots_probs_convergence():
    """With many shots, shot-sampled probs should converge to exact."""
    script = Script(param_bell_circuit, n_qubits=2)
    key = jax.random.PRNGKey(123)
    exact = script.execute(type="probs", args=(0.5,))
    sampled = script.execute(type="probs", args=(0.5,), shots=100000, key=key)
    assert jnp.allclose(exact, sampled, atol=0.02), (
        f"Shot probs don't converge to exact.\n"
        f"  exact:   {exact}\n"
        f"  sampled: {sampled}"
    )


@pytest.mark.unittest
def test_shots_expval_single():
    """Shot-sampled expval should be close to exact for many shots."""
    script = Script(param_bell_circuit, n_qubits=2)
    key = jax.random.PRNGKey(7)
    obs = [PauliZ(wires=0), PauliZ(wires=1)]
    exact = script.execute(type="expval", obs=obs, args=(0.5,))
    sampled = script.execute(type="expval", obs=obs, args=(0.5,), shots=100000, key=key)
    assert (
        sampled.shape == exact.shape
    ), f"Shape mismatch: {sampled.shape} vs {exact.shape}"
    assert jnp.allclose(exact, sampled, atol=0.02), (
        f"Shot expval doesn't converge to exact.\n"
        f"  exact:   {exact}\n"
        f"  sampled: {sampled}"
    )


@pytest.mark.unittest
def test_shots_expval_bounded():
    """Shot-sampled expval for PauliZ should be in [-1, 1]."""
    script = Script(param_bell_circuit, n_qubits=2)
    key = jax.random.PRNGKey(99)
    obs = [PauliZ(wires=0)]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        result = script.execute(
            type="expval", obs=obs, args=(0.5,), shots=100, key=subkey
        )
        assert -1.0 <= float(result[0]) <= 1.0, f"Expval out of bounds: {result[0]}"


@pytest.mark.unittest
def test_shots_different_keys_give_different_results():
    """Different PRNG keys should produce different shot samples."""
    script = Script(param_bell_circuit, n_qubits=2)
    r1 = script.execute(
        type="probs",
        args=(0.5,),
        shots=100,
        key=jax.random.PRNGKey(0),
    )
    r2 = script.execute(
        type="probs",
        args=(0.5,),
        shots=100,
        key=jax.random.PRNGKey(1),
    )
    # With only 100 shots, different keys should almost always differ
    assert not jnp.allclose(r1, r2), "Different keys produced identical results"


@pytest.mark.unittest
def test_shots_probs_batched():
    """Shot-sampled probs with batched execution."""
    script = Script(param_bell_circuit, n_qubits=2)
    thetas = jnp.array([0.1, 0.5, 1.0, 2.0])
    key = jax.random.PRNGKey(42)
    result = script.execute(
        type="probs",
        args=(thetas,),
        in_axes=(0,),
        shots=10000,
        key=key,
    )
    assert result.shape == (4, 4), f"Expected shape (4, 4), got {result.shape}"
    # Each row should sum to 1
    row_sums = result.sum(axis=1)
    assert jnp.allclose(
        row_sums, 1.0, atol=1e-10
    ), f"Batched probs don't sum to 1: {row_sums}"


@pytest.mark.unittest
def test_shots_expval_batched():
    """Shot-sampled expval with batched execution converges to exact."""
    script = Script(param_bell_circuit, n_qubits=2)
    thetas = jnp.array([0.1, 0.5, 1.0])
    obs = [PauliZ(wires=0)]
    key = jax.random.PRNGKey(42)
    exact = script.execute(
        type="expval",
        obs=obs,
        args=(thetas,),
        in_axes=(0,),
    )
    sampled = script.execute(
        type="expval",
        obs=obs,
        args=(thetas,),
        in_axes=(0,),
        shots=100000,
        key=key,
    )
    assert sampled.shape == exact.shape
    assert jnp.allclose(exact, sampled, atol=0.02), (
        f"Batched shot expval doesn't converge.\n"
        f"  exact:   {exact}\n"
        f"  sampled: {sampled}"
    )


@pytest.mark.unittest
def test_shots_none_returns_exact():
    """shots=None should return exact analytic results (no sampling)."""
    script = Script(param_bell_circuit, n_qubits=2)
    r1 = script.execute(type="probs", args=(0.5,))
    r2 = script.execute(type="probs", args=(0.5,), shots=None)
    assert jnp.allclose(r1, r2), "shots=None should match exact results"


@pytest.mark.unittest
def test_shots_state_type_ignored():
    """shots parameter should be ignored for 'state' measurement type."""
    script = Script(param_bell_circuit, n_qubits=2)
    # For 'state' type, shots should have no effect (exact statevector)
    exact = script.execute(type="state", args=(0.5,))
    with_shots = script.execute(
        type="state",
        args=(0.5,),
        shots=100,
        key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(exact, with_shots), "shots should be ignored for 'state' type"


@pytest.mark.unittest
def test_dagger():
    def circuit():
        RX(0.5, wires=0)
        RX(0.5, wires=0).dagger()

    obs = [PauliZ(0)]
    script = Script(circuit)
    res = script.execute(type="expval", obs=obs)
    assert jnp.allclose(res, 1), "Dagger should undo operation"


@pytest.mark.unittest
def test_power():
    def circuit():
        PauliX(wires=0).power(2)

    obs = [PauliZ(0)]
    script = Script(circuit)
    res = script.execute(type="expval", obs=obs)
    assert jnp.allclose(res, 1), "Dagger should undo operation"


@pytest.mark.unittest
def test_estimate_peak_bytes_basic():
    """Memory estimates should be positive and scale with batch size."""
    est1 = Script._estimate_peak_bytes(5, 1, "state", False)
    est100 = Script._estimate_peak_bytes(5, 100, "state", False)
    assert est1 > 0
    assert est100 > est1
    # Should scale roughly linearly (within safety factor tolerance)
    assert est100 <= est1 * 200  # generous upper bound


@pytest.mark.unittest
def test_estimate_peak_bytes_density_larger():
    """Density mode should estimate more memory than state mode."""
    est_state = Script._estimate_peak_bytes(5, 100, "state", False)
    est_density = Script._estimate_peak_bytes(5, 100, "density", True)
    assert est_density > est_state


@pytest.mark.unittest
def test_estimate_peak_bytes_qubits_scaling():
    """Memory should scale exponentially with qubit count."""
    est4 = Script._estimate_peak_bytes(4, 10, "state", False)
    est8 = Script._estimate_peak_bytes(8, 10, "state", False)
    # 8 qubits: dim=256 vs 4 qubits: dim=16  → ~16× more
    assert est8 > est4 * 10


@pytest.mark.unittest
def test_available_memory_bytes():
    """Available memory should return a positive value."""
    avail = Script._available_memory_bytes()
    assert avail > 0
    # Should be at least 2 GB on any reasonable system
    assert avail >= 2 * 1024**3


@pytest.mark.unittest
def test_compute_chunk_size_fits():
    """When everything fits, chunk_size == batch_size."""
    # 2 qubits, 10 batch, state mode — tiny, always fits
    chunk = Script._compute_chunk_size(2, 10, "state", False)
    assert chunk == 10


@pytest.mark.unittest
def test_compute_chunk_size_too_large():
    """When batch doesn't fit, chunk_size < batch_size."""
    # Simulate a scenario that would exceed memory by using an absurd
    # qubit count — 30 qubits × 1M batch × density mode
    chunk = Script._compute_chunk_size(
        n_qubits=30,
        batch_size=1000000,
        type="density",
        use_density=True,
        n_obs=0,
    )
    assert chunk < 1000000, f"Expected chunk < 1000000, got {chunk}"
    assert chunk >= 1, f"Chunk size must be at least 1, got {chunk}"


@pytest.mark.unittest
def test_compute_chunk_size_minimum_one():
    """Chunk size should never be less than 1."""
    chunk = Script._compute_chunk_size(
        n_qubits=30,
        batch_size=100,
        type="density",
        use_density=True,
        n_obs=0,
        memory_fraction=0.00001,
    )
    assert chunk >= 1


@pytest.mark.unittest
def test_chunked_execution_matches_full():
    """Chunked execution should produce the same results as full batch."""

    def circuit(theta):
        RX(theta, wires=0)
        CX(wires=[0, 1])

    script = Script(circuit, n_qubits=2)
    thetas = jnp.linspace(0, jnp.pi, 20)
    obs = [PauliZ(wires=0), PauliZ(wires=1)]

    # Full batch execution (no chunking)
    full_result = script.execute(
        type="expval",
        obs=obs,
        args=(thetas,),
        in_axes=(0,),
    )

    # Force chunked execution by calling _execute_chunked directly
    # First, build the vmapped function
    script2 = Script(circuit, n_qubits=2)
    # Run once to populate cache
    _ = script2.execute(
        type="expval",
        obs=obs,
        args=(thetas,),
        in_axes=(0,),
    )

    # Retrieve cached function
    from qml_essentials.gates import UnitaryGates

    arg_shapes = tuple(
        (a.shape, a.dtype) if hasattr(a, "shape") else type(a) for a in (thetas,)
    )
    cache_key = ("expval", (0,), arg_shapes, UnitaryGates.batch_gate_error)
    batched_fn, _, _ = script2._jit_cache[cache_key]

    # Now execute with chunk_size=5 (4 chunks of 5)
    chunked_result = Script._execute_chunked(
        batched_fn, (thetas,), (0,), batch_size=20, chunk_size=5
    )

    assert jnp.allclose(full_result, chunked_result, atol=1e-10), (
        f"Chunked results don't match full batch.\n"
        f"  full:    {full_result}\n"
        f"  chunked: {chunked_result}"
    )


@pytest.mark.unittest
def test_chunked_probs_matches_full():
    """Chunked probs execution should match full batch."""

    def circuit(theta):
        RX(theta, wires=0)
        H(wires=1)

    script = Script(circuit, n_qubits=2)
    thetas = jnp.linspace(0, jnp.pi, 12)

    # Full batch
    full_result = script.execute(
        type="probs",
        args=(thetas,),
        in_axes=(0,),
    )

    # Chunked
    script2 = Script(circuit, n_qubits=2)
    _ = script2.execute(type="probs", args=(thetas,), in_axes=(0,))

    from qml_essentials.gates import UnitaryGates

    arg_shapes = tuple(
        (a.shape, a.dtype) if hasattr(a, "shape") else type(a) for a in (thetas,)
    )
    cache_key = ("probs", (0,), arg_shapes, UnitaryGates.batch_gate_error)
    batched_fn, _, _ = script2._jit_cache[cache_key]

    chunked_result = Script._execute_chunked(
        batched_fn, (thetas,), (0,), batch_size=12, chunk_size=4
    )

    assert full_result.shape == chunked_result.shape
    assert jnp.allclose(full_result, chunked_result, atol=1e-10)


@pytest.mark.unittest
def test_chunked_density_matches_full():
    """Chunked density execution should match full batch."""

    def circuit(theta):
        RX(theta, wires=0)

    script = Script(circuit, n_qubits=1)
    thetas = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    full_result = script.execute(
        type="density",
        args=(thetas,),
        in_axes=(0,),
    )

    script2 = Script(circuit, n_qubits=1)
    _ = script2.execute(type="density", args=(thetas,), in_axes=(0,))

    from qml_essentials.gates import UnitaryGates

    arg_shapes = tuple(
        (a.shape, a.dtype) if hasattr(a, "shape") else type(a) for a in (thetas,)
    )
    cache_key = ("density", (0,), arg_shapes, UnitaryGates.batch_gate_error)
    batched_fn, _, _ = script2._jit_cache[cache_key]

    chunked_result = Script._execute_chunked(
        batched_fn, (thetas,), (0,), batch_size=6, chunk_size=2
    )

    assert full_result.shape == chunked_result.shape
    assert jnp.allclose(full_result, chunked_result, atol=1e-10)


@pytest.mark.unittest
def test_chunked_uneven_batch():
    """Chunking should work when batch_size is not divisible by chunk_size."""

    def circuit(theta):
        RX(theta, wires=0)

    script = Script(circuit, n_qubits=1)
    thetas = jnp.linspace(0, jnp.pi, 7)  # 7 elements

    full_result = script.execute(
        type="probs",
        args=(thetas,),
        in_axes=(0,),
    )

    script2 = Script(circuit, n_qubits=1)
    _ = script2.execute(type="probs", args=(thetas,), in_axes=(0,))

    from qml_essentials.gates import UnitaryGates

    arg_shapes = tuple(
        (a.shape, a.dtype) if hasattr(a, "shape") else type(a) for a in (thetas,)
    )
    cache_key = ("probs", (0,), arg_shapes, UnitaryGates.batch_gate_error)
    batched_fn, _, _ = script2._jit_cache[cache_key]

    # 7 elements, chunk_size=3 → chunks of [3, 3, 1]
    chunked_result = Script._execute_chunked(
        batched_fn, (thetas,), (0,), batch_size=7, chunk_size=3
    )

    assert chunked_result.shape == full_result.shape
    assert jnp.allclose(full_result, chunked_result, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_statevector_identical():
    """Fidelity of identical state vectors should be 1."""
    sv = jnp.array([1.0, 0.0])
    result = fidelity(sv, sv)
    expected = qml.math.fidelity_statevector(np.array(sv), np.array(sv))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, 1.0, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_statevector_orthogonal():
    """Fidelity of orthogonal state vectors should be 0."""
    sv0 = jnp.array([1.0, 0.0])
    sv1 = jnp.array([0.0, 1.0])
    result = fidelity(sv0, sv1)
    expected = qml.math.fidelity_statevector(np.array(sv0), np.array(sv1))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, 0.0, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_statevector_overlap():
    """Fidelity of two arbitrary pure states should match PennyLane."""
    sv0 = jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j])
    sv1 = jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j])
    result = fidelity(sv0, sv1)
    expected = qml.math.fidelity_statevector(np.array(sv0), np.array(sv1))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_statevector_batched():
    """Batched state-vector fidelity should match element-wise PennyLane results."""
    sv0_batch = jnp.array([[1.0, 0.0], [0.0, 1.0], [1 / jnp.sqrt(2), 1 / jnp.sqrt(2)]])
    sv1_batch = jnp.array([[1.0, 0.0], [1.0, 0.0], [1 / jnp.sqrt(2), -1 / jnp.sqrt(2)]])
    result = fidelity(sv0_batch, sv1_batch)
    for i in range(3):
        expected_i = qml.math.fidelity_statevector(
            np.array(sv0_batch[i]), np.array(sv1_batch[i])
        )
        assert jnp.allclose(result[i], expected_i, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_dm_identical():
    """Fidelity of identical density matrices should be 1."""
    rho = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    result = fidelity(rho, rho)
    expected = qml.math.fidelity(np.array(rho), np.array(rho))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, 1.0, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_dm_orthogonal():
    """Fidelity of orthogonal pure-state density matrices should be 0."""
    rho0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    rho1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)
    result = fidelity(rho0, rho1)
    expected = qml.math.fidelity(np.array(rho0), np.array(rho1))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, 0.0, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_dm_mixed():
    """Fidelity between a pure state and the maximally mixed state."""
    rho0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    rho_mixed = jnp.eye(2, dtype=jnp.complex128) / 2
    result = fidelity(rho0, rho_mixed)
    expected = qml.math.fidelity(np.array(rho0), np.array(rho_mixed))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_dm_batched():
    """Batched density-matrix fidelity should match element-wise PennyLane."""
    rho0_batch = jnp.array(
        [
            [[1, 0], [0, 0]],
            [[0.5, 0.5], [0.5, 0.5]],
            jnp.eye(2) / 2,
        ],
        dtype=jnp.complex128,
    )
    rho1_batch = jnp.array(
        [
            [[0, 0], [0, 1]],
            [[0.5, 0.5], [0.5, 0.5]],
            jnp.eye(2) / 2,
        ],
        dtype=jnp.complex128,
    )
    result = fidelity(rho0_batch, rho1_batch)
    for i in range(3):
        expected_i = qml.math.fidelity(np.array(rho0_batch[i]), np.array(rho1_batch[i]))
        assert jnp.allclose(result[i], expected_i, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_dm_matches_statevector():
    """Density-matrix fidelity of pure states should equal state-vector fidelity."""
    sv0 = jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j])
    sv1 = jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j])
    rho0 = jnp.outer(sv0, jnp.conj(sv0))
    rho1 = jnp.outer(sv1, jnp.conj(sv1))
    f_sv = fidelity(sv0, sv1)
    f_dm = fidelity(rho0, rho1)
    assert jnp.allclose(f_sv, f_dm, atol=1e-8)


@pytest.mark.unittest
def test_fidelity_mismatched_raises():
    """Passing a vector and a matrix should raise ValueError."""
    sv = jnp.array([1.0, 0.0])
    dm = jnp.eye(2, dtype=jnp.complex128)
    with pytest.raises(ValueError, match="same kind"):
        fidelity(sv, dm)


@pytest.mark.unittest
def test_trace_distance_identical():
    """Trace distance of identical states should be 0."""
    rho = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    result = trace_distance(rho, rho)
    expected = qml.math.trace_distance(np.array(rho), np.array(rho))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, 0.0, atol=1e-10)


@pytest.mark.unittest
def test_trace_distance_orthogonal():
    """Trace distance of orthogonal pure states should be 1."""
    rho0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    rho1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)
    result = trace_distance(rho0, rho1)
    expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, 1.0, atol=1e-10)


@pytest.mark.unittest
def test_trace_distance_mixed():
    """Trace distance between a pure state and the maximally mixed state."""
    rho0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    rho_mixed = jnp.eye(2, dtype=jnp.complex128) / 2
    result = trace_distance(rho0, rho_mixed)
    expected = qml.math.trace_distance(np.array(rho0), np.array(rho_mixed))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_trace_distance_batched():
    """Batched trace distance should match element-wise PennyLane."""
    batch0 = jnp.array(
        [jnp.eye(2) / 2, jnp.ones((2, 2)) / 2, jnp.array([[1, 0], [0, 0]])],
        dtype=jnp.complex128,
    )
    batch1 = jnp.array(
        [jnp.ones((2, 2)) / 2, jnp.ones((2, 2)) / 2, jnp.array([[1, 0], [0, 0]])],
        dtype=jnp.complex128,
    )
    result = trace_distance(batch0, batch1)
    expected = qml.math.trace_distance(np.array(batch0), np.array(batch1))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_trace_distance_from_statevectors():
    """Trace distance computed from outer-product DMs should match PennyLane."""
    sv0 = jnp.array([0.2, jnp.sqrt(0.96)])
    sv1 = jnp.array([1.0, 0.0])
    rho0 = jnp.outer(sv0, jnp.conj(sv0))
    rho1 = jnp.outer(sv1, jnp.conj(sv1))
    result = trace_distance(rho0, rho1)
    expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_fidelity_2qubit_statevector():
    """Fidelity of 2-qubit state vectors should match PennyLane."""
    sv0 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])  # Bell state
    sv1 = jnp.array([1, 0, 0, 0], dtype=jnp.complex128)  # |00⟩
    result = fidelity(sv0, sv1)
    expected = qml.math.fidelity_statevector(np.array(sv0), np.array(sv1))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_trace_distance_2qubit():
    """Trace distance of 2-qubit density matrices should match PennyLane."""
    bell = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])
    rho0 = jnp.outer(bell, jnp.conj(bell))
    rho1 = jnp.eye(4, dtype=jnp.complex128) / 4  # maximally mixed
    result = trace_distance(rho0, rho1)
    expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_phase_difference_identical():
    """Phase difference of identical states should be |1 - 0| = 1."""
    sv = jnp.array([1.0, 0.0])
    result = phase_difference(sv, sv)
    # angle(⟨ψ|ψ⟩) = angle(1) = 0, so |1 - 0| = 1
    expected = jnp.abs(1.0 - jnp.angle(jnp.vdot(sv, sv)))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, 1.0, atol=1e-10)


@pytest.mark.unittest
def test_phase_difference_global_phase():
    """A global phase of pi should give |1 - pi|."""
    sv0 = jnp.array([1.0, 0.0])
    sv1 = jnp.array([-1.0, 0.0])  # global phase of pi
    result = phase_difference(sv0, sv1)
    expected = jnp.abs(1.0 - jnp.angle(jnp.vdot(sv0, sv1)))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, jnp.abs(1.0 - jnp.pi), atol=1e-10)


@pytest.mark.unittest
def test_phase_difference_half_pi():
    """A global phase of pi/2 should give |1 - pi/2|."""
    sv0 = jnp.array([1.0, 0.0])
    sv1 = jnp.array([1j, 0.0])  # global phase of pi/2
    result = phase_difference(sv0, sv1)
    expected = jnp.abs(1.0 - jnp.angle(jnp.vdot(sv0, sv1)))
    assert jnp.allclose(result, expected, atol=1e-10)
    assert jnp.allclose(result, jnp.abs(1.0 - jnp.pi / 2), atol=1e-10)


@pytest.mark.unittest
def test_phase_difference_arbitrary():
    """Phase difference of two arbitrary states should match the vdot formula."""
    sv0 = jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j])
    sv1 = jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j])
    result = phase_difference(sv0, sv1)
    expected = float(jnp.abs(1.0 - jnp.angle(jnp.vdot(sv0, sv1))))
    assert jnp.allclose(result, expected, atol=1e-10)


@pytest.mark.unittest
def test_phase_difference_batched():
    """Batched phase difference should match element-wise computation."""
    sv0_batch = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1 / jnp.sqrt(2), 1 / jnp.sqrt(2)],
        ]
    )
    sv1_batch = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1 / jnp.sqrt(2), -1 / jnp.sqrt(2)],
        ]
    )
    result = phase_difference(sv0_batch, sv1_batch)
    for i in range(3):
        inner = jnp.sum(jnp.conj(sv0_batch[i]) * sv1_batch[i])
        expected_i = float(jnp.abs(1.0 - jnp.angle(inner)))
        assert jnp.allclose(result[i], expected_i, atol=1e-10)


@pytest.mark.unittest
def test_phase_difference_2qubit():
    """Phase difference of 2-qubit states should match the vdot formula."""
    sv0 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])  # Bell |Φ+⟩
    sv1 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1j / jnp.sqrt(2)])  # phase on |11⟩
    result = phase_difference(sv0, sv1)
    expected = float(jnp.abs(1.0 - jnp.angle(jnp.vdot(sv0, sv1))))
    assert jnp.allclose(result, expected, atol=1e-10)
