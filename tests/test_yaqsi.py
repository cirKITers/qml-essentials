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
    Operation,
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
    PauliY,
    PauliZ,
    Id,
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
from qml_essentials.gates import (
    PulseEnvelope,
    PulseInformation,
    PulseGates,
)

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
    H(wires=0)
    CX(wires=[0, 1])
    CX(wires=[1, 2])


def ghz_circuit_4(*args, **kwargs):
    H(wires=0)
    CX(wires=[0, 1])
    CX(wires=[1, 2])
    CX(wires=[2, 3])


def ghz_toffoli_3(*args, **kwargs):
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
def test_jax_gradient() -> None:
    """Gradient of ⟨Z⟩ w.r.t. θ for RX(θ)|0⟩ equals -sin(θ)."""
    script = Script(f=parametrized_circuit)
    theta_val = jnp.array(0.5)

    def cost(theta):
        return script.execute(type="expval", obs=[PauliZ(0)], args=(theta,))[0]

    grad_fn = jax.grad(cost)
    expected_grad = -jnp.sin(theta_val)
    assert jnp.allclose(grad_fn(theta_val), expected_grad, atol=1e-6)


class TestEvolve:
    @pytest.mark.unittest
    def test_evolve_from_zero(self) -> None:
        """exp(-i·t·Z)|0⟩ leaves ⟨X⟩ = 0 since |0⟩ is a Z eigenstate."""
        script = Script(f=evol_circuit)
        res = script.execute(type="expval", obs=[PauliX(0)], args=(0.3,))
        assert jnp.allclose(res, jnp.array([0.0]), atol=1e-6)

    @pytest.mark.unittest
    def test_evolve_from_plus(self) -> None:
        """exp(-i·t·Z)|+⟩ gives |⟨X⟩| = cos(2t)."""
        t = 0.3
        script = Script(f=evol_circuit_plus)
        res = script.execute(type="expval", obs=[PauliX(0)], args=(t,))
        assert jnp.allclose(jnp.abs(res), jnp.cos(2 * t), atol=1e-6)

    @pytest.mark.unittest
    def test_evolve_parametrized_constant_matches_static(self) -> None:
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
    def test_evolve_parametrized_unitarity(self) -> None:
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
    def test_evolve_parametrized_differentiable(self) -> None:
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
    def test_evolve_parametrized_on_tape(self) -> None:
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
    def test_evolve_parametrized_hermitian_not_on_tape(self) -> None:
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
    def test_evolve_type_error(self) -> None:
        """evolve() with an unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="evolve"):
            evolve("not a hamiltonian")

    @pytest.mark.unittest
    def test_evolve_max_steps_throws_by_default(self) -> None:
        """A tight ``max_steps`` budget on a fast-oscillating Hamiltonian
        triggers a solver error (default ``throw=True``)."""
        from qml_essentials.yaqsi import Yaqsi

        # Highly oscillatory coefficient — Tsit5 will need many steps
        def fast_coeff(p, t):
            return p[0] * jnp.cos(1.0e3 * t)

        Z = PauliZ._matrix
        ph = fast_coeff * Hermitian(matrix=Z, wires=0, record=False)

        prev = Yaqsi.set_solver_defaults(max_steps=4, throw=True)
        try:
            with pytest.raises(Exception):
                evolve(ph)([jnp.array([1.0])], 1.0)
        finally:
            Yaqsi.set_solver_defaults(**prev)

    @pytest.mark.unittest
    def test_evolve_throw_false_returns_nan_on_failure(self) -> None:
        """With ``throw=False`` and an unreachable budget, a failed solve
        returns a NaN-filled unitary instead of raising."""
        from qml_essentials.yaqsi import Yaqsi

        def fast_coeff(p, t):
            return p[0] * jnp.cos(1.0e3 * t)

        Z = PauliZ._matrix
        ph = fast_coeff * Hermitian(matrix=Z, wires=0, record=False)

        prev = Yaqsi.set_solver_defaults(max_steps=4, throw=False)
        try:
            op_ = evolve(ph)([jnp.array([1.0])], 1.0)
            U = op_.matrix
            assert jnp.all(jnp.isnan(U)), (
                "Expected NaN-filled unitary on solver failure with "
                f"throw=False, got\n{U}"
            )
        finally:
            Yaqsi.set_solver_defaults(**prev)

    @pytest.mark.unittest
    def test_evolve_throw_false_succeeds_on_easy_problem(self) -> None:
        """``throw=False`` does not affect well-behaved problems: the
        returned unitary is finite and equals the static-evolve result
        for a constant coefficient."""
        from qml_essentials.yaqsi import Yaqsi

        def const_coeff(p, t):
            return 1.0

        Z = PauliZ._matrix
        ph = const_coeff * Hermitian(matrix=Z, wires=0, record=False)

        prev = Yaqsi.set_solver_defaults(throw=False)
        try:
            U = evolve(ph)([jnp.array([0.0])], 0.5).matrix
        finally:
            Yaqsi.set_solver_defaults(**prev)

        U_static = jax.scipy.linalg.expm(-1j * 0.5 * Z)
        assert jnp.all(jnp.isfinite(U))
        assert jnp.allclose(U, U_static, atol=1e-6)


class TestMeasurement:
    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "obs_cls",
        [PauliX, PauliZ],
        ids=["X", "Z"],
    )
    def test_expval_bell(self, obs_cls) -> None:
        """Bell state (|00⟩+|11⟩)/√2 has ⟨Oᵢ⟩ = 0 for O ∈ {X, Z}."""
        script = Script(f=bell_circuit)
        res = script.execute(type="expval", obs=[obs_cls(0), obs_cls(1)])
        assert jnp.allclose(res, jnp.array([0.0, 0.0]), atol=1e-10)

    @pytest.mark.unittest
    def test_probs_bell(self) -> None:
        """Bell state yields |00⟩ and |11⟩ each with probability 0.5."""
        script = Script(f=bell_circuit)
        probs = script.execute(type="probs")
        assert jnp.allclose(probs, jnp.array([0.5, 0.0, 0.0, 0.5]), atol=1e-10)

    @pytest.mark.unittest
    def test_parametrized_expval(self) -> None:
        """RX(θ)|0⟩ gives ⟨Z⟩ = cos(θ)."""
        script = Script(f=parametrized_circuit)
        theta_val = jnp.array(0.5)

        def cost(theta):
            return script.execute(type="expval", obs=[PauliZ(0)], args=(theta,))[0]

        assert jnp.allclose(cost(theta_val), jnp.cos(theta_val), atol=1e-6)

    @pytest.mark.unittest
    def test_probs_ghz_3(self) -> None:
        """3-qubit GHZ state has prob 0.5 on |000⟩ and |111⟩, zero elsewhere."""
        script = Script(f=ghz_circuit_3)
        probs = script.execute(type="probs")
        expected = jnp.zeros(8).at[0].set(0.5).at[7].set(0.5)
        assert jnp.allclose(probs, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_expval_ghz_3_z(self) -> None:
        """Each qubit of a 3-qubit GHZ state has ⟨Zᵢ⟩ = 0."""
        script = Script(f=ghz_circuit_3)
        obs = [PauliZ(0), PauliZ(1), PauliZ(2)]
        res = script.execute(type="expval", obs=obs)
        assert jnp.allclose(res, jnp.zeros(3), atol=1e-10)

    @pytest.mark.unittest
    def test_probs_ghz_4(self) -> None:
        """4-qubit GHZ state has prob 0.5 on |0000⟩ and |1111⟩, zero elsewhere."""
        script = Script(f=ghz_circuit_4)
        probs = script.execute(type="probs")
        expected = jnp.zeros(16).at[0].set(0.5).at[15].set(0.5)
        assert jnp.allclose(probs, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_probs_ghz_toffoli(self) -> None:
        """3-qubit GHZ via Toffoli matches the CNOT-chain result."""
        script_cnot = Script(f=ghz_circuit_3)
        script_toff = Script(f=ghz_toffoli_3)
        probs_cnot = script_cnot.execute(type="probs")
        probs_toff = script_toff.execute(type="probs")
        assert jnp.allclose(probs_cnot, probs_toff, atol=1e-10)

    @pytest.mark.unittest
    def test_state_ghz_3_non_adjacent_wires(self) -> None:
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
    def test_density_pure_state_is_projector(self) -> None:
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
    def test_density_bell_matches_statevector(self) -> None:
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
    def test_density_diagonal_equals_probs(self) -> None:
        """
        The diagonal of the density matrix equals the probability vector
        from the statevector path, for any pure circuit.
        """
        script = Script(f=ghz_circuit_3)
        probs = script.execute(type="probs")
        rho = script.execute(type="density")

        assert jnp.allclose(jnp.real(jnp.diag(rho)), probs, atol=1e-10)

    @pytest.mark.unittest
    def test_density_is_hermitian(self) -> None:
        """Density matrix must satisfy ρ = ρ†."""
        script = Script(f=ghz_circuit_4)
        rho = script.execute(type="density")

        assert jnp.allclose(rho, jnp.conj(rho.T), atol=1e-10)

    @pytest.mark.unittest
    def test_density_expval_matches_statevector(self) -> None:
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


class TestPennylane:

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "gate_name,yaqsi_gate,pl_gate,theta,prep_both",
        [
            ("CY", CY, qml.CY, None, False),
            ("CZ", CZ, qml.CZ, None, True),
            ("CRX", CRX, qml.CRX, 1.3, False),
            ("CRY", CRY, qml.CRY, 0.9, False),
            ("CRZ", CRZ, qml.CRZ, 2.1, False),
        ],
        ids=["CY", "CZ", "CRX", "CRY", "CRZ"],
    )
    def test_controlled_gate_matches_pennylane(
        self, gate_name, yaqsi_gate, pl_gate, theta, prep_both
    ) -> None:
        """Controlled gate probabilities match PennyLane."""

        def yaqsi_circuit():
            H(wires=0)
            if prep_both:
                H(wires=1)
            if theta is not None:
                yaqsi_gate(theta, wires=[0, 1])
            else:
                yaqsi_gate(wires=[0, 1])

        def pl_circuit():
            qml.Hadamard(wires=0)
            if prep_both:
                qml.Hadamard(wires=1)
            if theta is not None:
                pl_gate(theta, wires=[0, 1])
            else:
                pl_gate(wires=[0, 1])

        script = Script(f=yaqsi_circuit)
        probs_ours = np.array(script.execute(type="probs"))
        probs_pl = _pennylane_probs(pl_circuit)

        assert np.allclose(
            probs_ours, probs_pl, atol=1e-10
        ), f"{gate_name} mismatch:\nours = {probs_ours}\nPL   = {probs_pl}"

    @pytest.mark.unittest
    def test_rot_matches_pennylane(self) -> None:
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
    def test_rot_decomposition_matches_individual_gates(self) -> None:
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


class TestNoise:
    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "channel_name,yaqsi_channel,pl_channel,param,theta,atol",
        [
            ("BitFlip", BitFlip, qml.BitFlip, 0.15, 0.8, 1e-8),
            ("PhaseFlip", PhaseFlip, qml.PhaseFlip, 0.2, 1.1, 1e-8),
            (
                "DepolarizingChannel",
                DepolarizingChannel,
                qml.DepolarizingChannel,
                0.12,
                0.6,
                1e-7,
            ),
            (
                "AmplitudeDamping",
                AmplitudeDamping,
                qml.AmplitudeDamping,
                0.25,
                1.3,
                1e-8,
            ),
            ("PhaseDamping", PhaseDamping, qml.PhaseDamping, 0.3, 0.9, 1e-8),
        ],
        ids=[
            "BitFlip",
            "PhaseFlip",
            "DepolarizingChannel",
            "AmplitudeDamping",
            "PhaseDamping",
        ],
    )
    def test_noise_channel_matches_pennylane(
        self, channel_name, yaqsi_channel, pl_channel, param, theta, atol
    ) -> None:
        """Noise channel density matrix matches PennyLane default.mixed."""

        def yaqsi_circuit(t):
            RX(t, wires=0)
            yaqsi_channel(param, wires=0)

        def pl_circuit():
            qml.RX(theta, wires=0)
            pl_channel(param, wires=0)

        script = Script(f=yaqsi_circuit)
        rho_ours = np.array(script.execute(type="density", args=(jnp.array(theta),)))
        rho_pl = _pennylane_density(pl_circuit)

        assert np.allclose(
            rho_ours, rho_pl, atol=atol
        ), f"{channel_name} mismatch:\nours =\n{rho_ours}\nPL =\n{rho_pl}"

    @pytest.mark.unittest
    def test_thermal_relaxation_t2_le_t1_matches_pennylane(self) -> None:
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
    def test_noise_auto_routes_to_density(self) -> None:
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
    def test_noise_density_is_valid(self) -> None:
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


class TestBatch:
    @pytest.mark.unittest
    def test_batched_expval_matches_sequential(self) -> None:
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
    def test_batched_expval_values(self) -> None:
        """
        RX(θ)|0⟩ -> ⟨Z⟩ = cos(θ) must hold element-wise across a batch.

        Mirrors the B_P (parameter-batch) axis from model.py where params
        has shape (B_P, n_layers, n_params) and in_axes=(0, ...).
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
    def test_batched_probs(self) -> None:
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
    def test_batched_gradient(self) -> None:
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
    def test_batched_broadcast_none_axis(self) -> None:
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
    def test_batched_in_axes_mismatch_raises(self) -> None:
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
    def test_batched_multi_qubit(self) -> None:
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
@pytest.mark.parametrize("keep", [[0], [1]], ids=["keep_0", "keep_1"])
def test_partial_trace_bell(keep) -> None:
    """Tracing out one qubit of the Bell state gives the maximally mixed state."""
    script = Script(f=bell_circuit)
    rho = script.execute(type="density")
    rho_reduced = partial_trace(rho, n_qubits=2, keep=keep)
    expected = 0.5 * jnp.eye(2, dtype=jnp.complex128)
    assert jnp.allclose(rho_reduced, expected, atol=1e-10)


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
    """Two-qubit parity observable is Z\\otimesZ."""
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
@pytest.mark.parametrize(
    "record,expected_len",
    [(False, 0), (True, 1)],
    ids=["record_false", "record_true"],
)
def test_hermitian_record_on_tape(record, expected_len) -> None:
    """Hermitian(record=...) should appear on the tape only when record=True."""
    from qml_essentials.tape import recording

    with recording() as tape:
        if record:
            _ = Hermitian(matrix=PauliZ._matrix, wires=0)
        else:
            _ = Hermitian(matrix=PauliZ._matrix, wires=0, record=False)
    assert len(tape) == expected_len


@pytest.mark.unittest
def test_parametrized_hamiltonian_creation() -> None:
    """coeff_fn * Hermitian produces a ParametrizedHamiltonian."""

    def coeff(p, t):
        return p * t

    herm = Hermitian(matrix=PauliZ._matrix, wires=0, record=False)
    ph = coeff * herm

    assert isinstance(ph, ParametrizedHamiltonian)
    assert ph.n_terms == 1
    assert ph.coeff_fns == (coeff,)
    assert jnp.allclose(ph.H_mats[0], PauliZ._matrix)
    assert ph.wires == [0]


@pytest.mark.unittest
def test_parametrized_hamiltonian_non_callable_raises() -> None:
    """Multiplying a non-callable with Hermitian raises TypeError."""
    herm = Hermitian(matrix=PauliZ._matrix, wires=0, record=False)
    with pytest.raises(TypeError, match="callable"):
        _ = 3.14 * herm


@pytest.mark.unittest
def test_parametrized_hamiltonian_multi_term_addition() -> None:
    """Adding two single-term PHs produces a 2-term PH with matching wires."""
    H_X = Hermitian(matrix=PauliX._matrix, wires=0, record=False)
    H_Y = Hermitian(matrix=PauliY._matrix, wires=0, record=False)

    def fx(p, t):
        return p[0]

    def fy(p, t):
        return p[0] * t

    ph = fx * H_X + fy * H_Y
    assert isinstance(ph, ParametrizedHamiltonian)
    assert ph.n_terms == 2
    assert ph.wires == [0]
    assert ph.coeff_fns == (fx, fy)


@pytest.mark.unittest
def test_parametrized_hamiltonian_wire_mismatch_raises() -> None:
    """Terms acting on different wires are rejected."""
    H_X0 = Hermitian(matrix=PauliX._matrix, wires=0, record=False)
    H_X1 = Hermitian(matrix=PauliX._matrix, wires=1, record=False)

    def f(p, t):
        return 1.0

    with pytest.raises(ValueError, match="same wires"):
        _ = (f * H_X0) + (f * H_X1)


@pytest.mark.unittest
def test_parametrized_hamiltonian_neg_and_sub() -> None:
    """-PH and PH - PH compose as expected."""
    H_X = Hermitian(matrix=PauliX._matrix, wires=0, record=False)

    def f(p, t):
        return 2.0

    ph = f * H_X
    ph_neg = -ph
    assert ph_neg.n_terms == 1
    # Coefficient should be negated
    assert jnp.allclose(ph_neg.coeff_fns[0](None, 0.0), -2.0)

    ph_sub = ph - ph  # should be a 2-term PH whose net coefficients cancel
    assert ph_sub.n_terms == 2
    c0 = ph_sub.coeff_fns[0](None, 0.0)
    c1 = ph_sub.coeff_fns[1](None, 0.0)
    assert jnp.allclose(c0 + c1, 0.0)


@pytest.mark.unittest
def test_evolve_multi_term_constant_matches_expm() -> None:
    """ODE under H = X + Y (constant) matches expm(-i(X+Y)*T)."""
    H_X = Hermitian(matrix=PauliX._matrix, wires=0, record=False)
    H_Y = Hermitian(matrix=PauliY._matrix, wires=0, record=False)

    def one_x(p, t):
        return 1.0

    def one_y(p, t):
        return 1.0

    ph = one_x * H_X + one_y * H_Y
    T = 0.5
    gate = evolve(ph)
    op = gate([jnp.array([]), jnp.array([])], T)
    U_ode = op.matrix

    H_sum = PauliX._matrix + PauliY._matrix
    U_ref = jax.scipy.linalg.expm(-1j * T * H_sum)
    assert jnp.allclose(U_ode, U_ref, atol=1e-6)


@pytest.mark.unittest
def test_evolve_multi_term_time_dependent_unitarity() -> None:
    """Time-dependent two-term Hamiltonian produces unitary evolution."""
    H_X = Hermitian(matrix=PauliX._matrix, wires=0, record=False)
    H_Y = Hermitian(matrix=PauliY._matrix, wires=0, record=False)

    def fx(p, t):
        return jnp.cos(2.0 * t)

    def fy(p, t):
        return -jnp.sin(2.0 * t)

    ph = fx * H_X + fy * H_Y
    gate = evolve(ph)
    op = gate([jnp.array([]), jnp.array([])], 0.7)
    U = op.matrix
    err = jnp.linalg.norm(U.conj().T @ U - jnp.eye(2))
    assert err < 1e-5


@pytest.mark.benchmark
@pytest.mark.unittest
@pytest.mark.parametrize(
    "mode,speedup", [("probs", 80), ("expval", 90), ("state", 70), ("density", 65)]
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


class TestShots:
    @pytest.mark.unittest
    def test_shots_probs_single(self):
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
    def test_shots_probs_convergence(self):
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
    def test_shots_expval_single(self):
        """Shot-sampled expval should be close to exact for many shots."""
        script = Script(param_bell_circuit, n_qubits=2)
        key = jax.random.PRNGKey(7)
        obs = [PauliZ(wires=0), PauliZ(wires=1)]
        exact = script.execute(type="expval", obs=obs, args=(0.5,))
        sampled = script.execute(
            type="expval", obs=obs, args=(0.5,), shots=100000, key=key
        )
        assert (
            sampled.shape == exact.shape
        ), f"Shape mismatch: {sampled.shape} vs {exact.shape}"
        assert jnp.allclose(exact, sampled, atol=0.02), (
            f"Shot expval doesn't converge to exact.\n"
            f"  exact:   {exact}\n"
            f"  sampled: {sampled}"
        )

    @pytest.mark.unittest
    def test_shots_expval_bounded(self):
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
    def test_shots_different_keys_give_different_results(self):
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
    def test_shots_probs_batched(self):
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
    def test_shots_expval_batched(self):
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
    def test_shots_none_returns_exact(self):
        """shots=None should return exact analytic results (no sampling)."""
        script = Script(param_bell_circuit, n_qubits=2)
        r1 = script.execute(type="probs", args=(0.5,))
        r2 = script.execute(type="probs", args=(0.5,), shots=None)
        assert jnp.allclose(r1, r2), "shots=None should match exact results"

    @pytest.mark.unittest
    def test_shots_state_type_ignored(self):
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
        assert jnp.allclose(
            exact, with_shots
        ), "shots should be ignored for 'state' type"


class TestGateOperations:
    @pytest.mark.unittest
    def test_dagger(self):
        def circuit():
            RX(0.5, wires=0)
            RX(0.5, wires=0).dagger()

        obs = [PauliZ(0)]
        script = Script(circuit)
        res = script.execute(type="expval", obs=obs)
        assert jnp.allclose(res, 1), "Dagger should undo operation"

    @pytest.mark.unittest
    def test_power(self):
        def circuit():
            PauliX(wires=0).power(2)

        obs = [PauliZ(0)]
        script = Script(circuit)
        res = script.execute(type="expval", obs=obs)
        assert jnp.allclose(res, 1), "Dagger should undo operation"

    @pytest.mark.unittest
    def test_mul_scalar_right(self):
        """PauliX * 2 should produce a matrix equal to 2 * X."""
        x = PauliX(wires=0, record=False)
        result = x * 2.0
        expected = 2.0 * PauliX._matrix
        assert jnp.allclose(result.matrix, expected, atol=1e-10)
        assert result.wires == [0]

    @pytest.mark.unittest
    def test_mul_scalar_left(self):
        """2 * PauliX should produce a matrix equal to 2 * X (rmul)."""
        x = PauliX(wires=0, record=False)
        result = 2.0 * x
        expected = 2.0 * PauliX._matrix
        assert jnp.allclose(result.matrix, expected, atol=1e-10)
        assert result.wires == [0]

    @pytest.mark.unittest
    def test_mul_updates_tape(self):
        """Scalar multiplication inside a circuit replaces the op on the tape."""
        from qml_essentials.tape import recording

        with recording() as tape:
            PauliX(wires=0) * 0.5

        assert len(tape) == 1, f"Expected 1 op on tape, got {len(tape)}"
        expected = 0.5 * PauliX._matrix
        assert jnp.allclose(tape[0].matrix, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_mul_in_circuit(self):
        """Scaled gate inside a circuit produces the expected expectation value.

        0.5 * X has eigenvalues ±0.5, so ⟨0| (0.5·X) |0⟩ should still be 0.
        """

        def circuit():
            PauliX(wires=0) * 0.5

        script = Script(circuit)
        # Use the scaled operator as observable
        obs = [Operation(wires=0, matrix=0.5 * PauliX._matrix, record=False)]
        res = script.execute(type="expval", obs=obs)
        assert jnp.allclose(res, jnp.array([0.0]), atol=1e-10)

    @pytest.mark.unittest
    def test_add_same_wires(self):
        """X + Z on the same wire should produce element-wise sum."""
        x = PauliX(wires=0, record=False)
        z = PauliZ(wires=0, record=False)
        result = x + z
        expected = PauliX._matrix + PauliZ._matrix
        assert jnp.allclose(result.matrix, expected, atol=1e-10)
        assert result.wires == [0]

    @pytest.mark.unittest
    def test_add_preserves_hermiticity(self):
        """Sum of two Hermitian operators is Hermitian: (X+Z)\\dagger = X+Z."""
        x = PauliX(wires=0, record=False)
        z = PauliZ(wires=0, record=False)
        result = x + z
        assert jnp.allclose(
            result.matrix, jnp.conj(result.matrix).T, atol=1e-10
        ), "Sum of Hermitian operators should be Hermitian"

    @pytest.mark.unittest
    def test_add_different_wires_raises(self):
        """Adding operations on different wires must raise ValueError."""
        x = PauliX(wires=0, record=False)
        z = PauliZ(wires=1, record=False)
        with pytest.raises(ValueError, match="same set of wires"):
            _ = x + z

    @pytest.mark.unittest
    def test_add_self(self):
        """X + X = 2 * X."""
        x = PauliX(wires=0, record=False)
        result = x + x
        expected = 2.0 * PauliX._matrix
        assert jnp.allclose(result.matrix, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_add_commutative(self):
        """Addition should be commutative: X + Y == Y + X."""
        x = PauliX(wires=0, record=False)
        y = PauliY(wires=0, record=False)
        assert jnp.allclose((x + y).matrix, (y + x).matrix, atol=1e-10)

    @pytest.mark.unittest
    def test_matmul_disjoint_wires(self):
        """X(0) \\otimes Z(1) produces a 4x4 Kronecker product on wires [0, 1]."""
        x = PauliX(wires=0, record=False)
        z = PauliZ(wires=1, record=False)
        result = x @ z
        expected = jnp.kron(PauliX._matrix, PauliZ._matrix)
        assert jnp.allclose(result.matrix, expected, atol=1e-10)
        assert result.wires == [0, 1]

    @pytest.mark.unittest
    def test_matmul_overlapping_wires_raises(self):
        """Tensor product with overlapping wires must raise ValueError."""
        x = PauliX(wires=0, record=False)
        z = PauliZ(wires=0, record=False)
        with pytest.raises(ValueError, match="overlapping wires"):
            _ = x @ z

    @pytest.mark.unittest
    def test_matmul_identity(self):
        """X(0) \\otimes I(1) should equal X \\otimes I."""
        x = PauliX(wires=0, record=False)
        i = Id(wires=1, record=False)
        result = x @ i
        expected = jnp.kron(PauliX._matrix, Id._matrix)
        assert jnp.allclose(result.matrix, expected, atol=1e-10)
        assert result.wires == [0, 1]

    @pytest.mark.unittest
    def test_matmul_dimension(self):
        """Tensor product of two 2x2 gates yields a 4x4 matrix."""
        x = PauliX(wires=0, record=False)
        y = PauliY(wires=1, record=False)
        result = x @ y
        assert result.matrix.shape == (4, 4)

    @pytest.mark.unittest
    def test_matmul_three_qubits(self):
        """X(0) \\otimes Y(1) \\otimes Z(2) yields an 8x8 matrix on wires [0, 1, 2]."""
        x = PauliX(wires=0, record=False)
        y = PauliY(wires=1, record=False)
        z = PauliZ(wires=2, record=False)
        result = (x @ y) @ z
        expected = jnp.kron(jnp.kron(PauliX._matrix, PauliY._matrix), PauliZ._matrix)
        assert jnp.allclose(result.matrix, expected, atol=1e-10)
        assert result.wires == [0, 1, 2]
        assert result.matrix.shape == (8, 8)


class TestMemory:
    @pytest.mark.unittest
    @pytest.mark.limit_memory("1 GB")
    def test_memory(self) -> None:
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

    @pytest.mark.unittest
    @pytest.mark.limit_memory("200 MB")
    def test_memory_statevector_probs(self) -> None:
        """Statevector probs for 12 qubits should stay well under 200 MB.

        Pure statevector simulation scales as O(2^n), so 12 qubits (dim=4096)
        with a batch of 10 should use only a few MB, not the O(4^n) that
        density-matrix simulation would require.
        """
        n_qubits = 12

        def circuit(theta):
            for i in range(n_qubits):
                H(wires=i)
            RX(theta, wires=0)

        script = Script(circuit, n_qubits=n_qubits)
        thetas = jnp.linspace(0, jnp.pi, 10)
        for _ in range(10):
            _ = script.execute(type="probs", args=(thetas,), in_axes=(0,))

    @pytest.mark.unittest
    @pytest.mark.limit_memory("200 MB")
    def test_memory_noisy_probs(self) -> None:
        """Noisy probs (density-matrix sim) for 8 qubits should stay under 200 MB.

        Even though density-matrix simulation is needed internally (O(4^n)),
        the returned output is only probabilities (O(2^n)).  This validates
        that intermediate density matrices are freed after measurement.
        """
        n_qubits = 8

        def circuit(theta):
            for i in range(n_qubits):
                H(wires=i)
            RX(theta, wires=0)
            BitFlip(0.01, wires=0)

        script = Script(circuit, n_qubits=n_qubits)
        thetas = jnp.linspace(0, jnp.pi, 5)
        for _ in range(5):
            result = script.execute(type="probs", args=(thetas,), in_axes=(0,))
            # Output should be probabilities, not density matrices
            assert result.shape == (5, 2**n_qubits)

    @pytest.mark.unittest
    @pytest.mark.limit_memory("200 MB")
    def test_memory_expval_scales_small(self) -> None:
        """Expval output should be tiny regardless of qubit count.

        For 12 qubits with 2 observables and batch=10, the output is only
        (10, 2) floats.  This validates that no density matrix is retained
        when computing expectation values on a pure circuit.
        """
        n_qubits = 12

        def circuit(theta):
            for i in range(n_qubits):
                H(wires=i)
            RX(theta, wires=0)

        script = Script(circuit, n_qubits=n_qubits)
        obs = [PauliZ(wires=0, record=False), PauliZ(wires=1, record=False)]
        thetas = jnp.linspace(0, jnp.pi, 10)
        for _ in range(10):
            result = script.execute(
                type="expval", obs=obs, args=(thetas,), in_axes=(0,)
            )
            assert result.shape == (10, 2)

    @pytest.mark.unittest
    def test_estimate_peak_bytes_basic(self):
        """Memory estimates should be positive and scale with batch size."""
        est1 = Script._estimate_peak_bytes(5, 1, "state", False)
        est100 = Script._estimate_peak_bytes(5, 100, "state", False)
        assert est1 > 0
        assert est100 > est1
        # Should scale roughly linearly (within safety factor tolerance)
        assert est100 <= est1 * 200  # generous upper bound

    @pytest.mark.unittest
    def test_estimate_peak_bytes_density_larger(self):
        """Density mode should estimate more memory than state mode."""
        est_state = Script._estimate_peak_bytes(5, 100, "state", False)
        est_density = Script._estimate_peak_bytes(5, 100, "density", True)
        assert est_density > est_state

    @pytest.mark.unittest
    def test_estimate_peak_bytes_qubits_scaling(self):
        """Memory should scale exponentially with qubit count."""
        est4 = Script._estimate_peak_bytes(4, 10, "state", False)
        est8 = Script._estimate_peak_bytes(8, 10, "state", False)
        # 8 qubits: dim=256 vs 4 qubits: dim=16  → ~16x more
        assert est8 > est4 * 10


class TestChunk:

    @pytest.mark.unittest
    @pytest.mark.limit_memory("1 GB")
    def test_memory_chunked_stays_bounded(self) -> None:
        """Chunked execution should not accumulate memory across chunks.

        Runs a 10-qubit density simulation with batch=20 chunked into
        size-5 sub-batches.  If chunk results were accumulated in a list,
        peak memory would be ~4x higher than a single chunk.  The
        pre-allocated output buffer approach should keep it bounded.

        We use a small initial batch (size=2) to populate the JIT cache
        without consuming much memory, then run the larger batch chunked.
        """
        n_qubits = 10

        def circuit(theta):
            for i in range(n_qubits):
                H(wires=i)
            RX(theta, wires=0)

        script = Script(circuit, n_qubits=n_qubits)

        # Populate the JIT cache with a small batch to avoid the large
        # full-batch allocation counting against our memory limit.
        small_thetas = jnp.array([0.0, 1.0])
        _ = script.execute(type="density", args=(small_thetas,), in_axes=(0,))

        # Retrieve the cached batched function
        from qml_essentials.gates import UnitaryGates

        arg_shapes = tuple(
            (a.shape, a.dtype) if hasattr(a, "shape") else type(a)
            for a in (small_thetas,)
        )
        cache_key = ("density", (0,), arg_shapes, (), UnitaryGates.batch_gate_error)
        batched_fn, _, _ = script._jit_cache[cache_key]

        # Now execute a larger batch in chunks of 5 (4 chunks total).
        # The JIT kernel is already compiled so no re-tracing occurs.
        thetas = jnp.linspace(0, jnp.pi, 20)
        result = Script._execute_chunked(
            batched_fn, (thetas,), (0,), batch_size=20, chunk_size=5
        )
        assert result.shape == (20, 2**n_qubits, 2**n_qubits)

    @pytest.mark.unittest
    def test_compute_chunk_size_fits(self):
        """When everything fits, chunk_size == batch_size."""
        # 2 qubits, 10 batch, state mode — tiny, always fits
        chunk = Script._compute_chunk_size(2, 10, "state", False)
        assert chunk == 10

    @pytest.mark.unittest
    def test_compute_chunk_size_too_large(self):
        """When batch doesn't fit, chunk_size < batch_size."""
        # Simulate a scenario that would exceed memory by using an absurd
        # qubit count — 30 qubits x 1M batch x density mode
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
    def test_compute_chunk_size_minimum_one(self):
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

    @staticmethod
    def _chunked_circuit_expval(theta):
        RX(theta, wires=0)
        CX(wires=[0, 1])

    @staticmethod
    def _chunked_circuit_probs(theta):
        RX(theta, wires=0)
        H(wires=1)

    @staticmethod
    def _chunked_circuit_density(theta):
        RX(theta, wires=0)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "circuit_fn,n_qubits,exec_type,thetas,chunk_size,obs",
        [
            (
                _chunked_circuit_expval,
                2,
                "expval",
                jnp.linspace(0, jnp.pi, 20),
                5,
                [PauliZ(wires=0), PauliZ(wires=1)],
            ),
            (
                _chunked_circuit_probs,
                2,
                "probs",
                jnp.linspace(0, jnp.pi, 12),
                4,
                None,
            ),
            (
                _chunked_circuit_density,
                1,
                "density",
                jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
                2,
                None,
            ),
        ],
        ids=["expval", "probs", "density"],
    )
    def test_chunked_matches_full(
        self, circuit_fn, n_qubits, exec_type, thetas, chunk_size, obs
    ):
        """Chunked execution should produce the same results as full batch."""
        exec_kwargs = dict(type=exec_type, args=(thetas,), in_axes=(0,))
        if obs is not None:
            exec_kwargs["obs"] = obs

        # Full batch execution
        script = Script(circuit_fn, n_qubits=n_qubits)
        full_result = script.execute(**exec_kwargs)

        # Build cached function for chunked execution
        script2 = Script(circuit_fn, n_qubits=n_qubits)
        _ = script2.execute(**exec_kwargs)

        from qml_essentials.gates import UnitaryGates

        arg_shapes = tuple(
            (a.shape, a.dtype) if hasattr(a, "shape") else type(a) for a in (thetas,)
        )
        cache_key = (exec_type, (0,), arg_shapes, (), UnitaryGates.batch_gate_error)
        batched_fn, _, _ = script2._jit_cache[cache_key]

        batch_size = thetas.shape[0]
        chunked_result = Script._execute_chunked(
            batched_fn, (thetas,), (0,), batch_size=batch_size, chunk_size=chunk_size
        )

        assert full_result.shape == chunked_result.shape
        assert jnp.allclose(full_result, chunked_result, atol=1e-10), (
            f"Chunked results don't match full batch.\n"
            f"  full:    {full_result}\n"
            f"  chunked: {chunked_result}"
        )

    @pytest.mark.unittest
    def test_chunked_uneven_batch(self):
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
        cache_key = ("probs", (0,), arg_shapes, (), UnitaryGates.batch_gate_error)
        batched_fn, _, _ = script2._jit_cache[cache_key]

        # 7 elements, chunk_size=3 → chunks of [3, 3, 1]
        chunked_result = Script._execute_chunked(
            batched_fn, (thetas,), (0,), batch_size=7, chunk_size=3
        )

        assert chunked_result.shape == full_result.shape
        assert jnp.allclose(full_result, chunked_result, atol=1e-10)


class TestFidelity:

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "sv0,sv1,expected_val",
        [
            (
                jnp.array([1.0, 0.0]),
                jnp.array([1.0, 0.0]),
                1.0,
            ),
            (
                jnp.array([1.0, 0.0]),
                jnp.array([0.0, 1.0]),
                0.0,
            ),
            (
                jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j]),
                jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j]),
                None,  # just check against PennyLane
            ),
        ],
        ids=["identical", "orthogonal", "overlap"],
    )
    def test_fidelity_statevector(self, sv0, sv1, expected_val):
        """Fidelity of state vectors should match PennyLane."""
        result = fidelity(sv0, sv1)
        expected = qml.math.fidelity_statevector(np.array(sv0), np.array(sv1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_statevector_batched(self):
        """Batched state-vector fidelity should match element-wise PennyLane results."""
        sv0_batch = jnp.array(
            [[1.0, 0.0], [0.0, 1.0], [1 / jnp.sqrt(2), 1 / jnp.sqrt(2)]]
        )
        sv1_batch = jnp.array(
            [[1.0, 0.0], [1.0, 0.0], [1 / jnp.sqrt(2), -1 / jnp.sqrt(2)]]
        )
        result = fidelity(sv0_batch, sv1_batch)
        for i in range(3):
            expected_i = qml.math.fidelity_statevector(
                np.array(sv0_batch[i]), np.array(sv1_batch[i])
            )
            assert jnp.allclose(result[i], expected_i, atol=1e-10)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "rho0,rho1,expected_val",
        [
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                1.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128),
                0.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.eye(2, dtype=jnp.complex128) / 2,
                None,  # just check against PennyLane
            ),
        ],
        ids=["identical", "orthogonal", "mixed"],
    )
    def test_fidelity_dm(self, rho0, rho1, expected_val):
        """Fidelity of density matrices should match PennyLane."""
        result = fidelity(rho0, rho1)
        expected = qml.math.fidelity(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_dm_batched(self):
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
            expected_i = qml.math.fidelity(
                np.array(rho0_batch[i]), np.array(rho1_batch[i])
            )
            assert jnp.allclose(result[i], expected_i, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_dm_matches_statevector(self):
        """Density-matrix fidelity of pure states should equal state-vector fidelity."""
        sv0 = jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j])
        sv1 = jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j])
        rho0 = jnp.outer(sv0, jnp.conj(sv0))
        rho1 = jnp.outer(sv1, jnp.conj(sv1))
        f_sv = fidelity(sv0, sv1)
        f_dm = fidelity(rho0, rho1)
        assert jnp.allclose(f_sv, f_dm, atol=1e-8)

    @pytest.mark.unittest
    def test_fidelity_mismatched_raises(self):
        """Passing a vector and a matrix should raise ValueError."""
        sv = jnp.array([1.0, 0.0])
        dm = jnp.eye(2, dtype=jnp.complex128)
        with pytest.raises(ValueError, match="same kind"):
            fidelity(sv, dm)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "rho0,rho1,expected_val",
        [
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                0.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128),
                1.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.eye(2, dtype=jnp.complex128) / 2,
                None,  # just check against PennyLane
            ),
        ],
        ids=["identical", "orthogonal", "mixed"],
    )
    def test_trace_distance(self, rho0, rho1, expected_val):
        """Trace distance should match PennyLane."""
        result = trace_distance(rho0, rho1)
        expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_trace_distance_batched(self):
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
    def test_trace_distance_from_statevectors(self):
        """Trace distance computed from outer-product DMs should match PennyLane."""
        sv0 = jnp.array([0.2, jnp.sqrt(0.96)])
        sv1 = jnp.array([1.0, 0.0])
        rho0 = jnp.outer(sv0, jnp.conj(sv0))
        rho1 = jnp.outer(sv1, jnp.conj(sv1))
        result = trace_distance(rho0, rho1)
        expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_2qubit_statevector(self):
        """Fidelity of 2-qubit state vectors should match PennyLane."""
        sv0 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])  # Bell state
        sv1 = jnp.array([1, 0, 0, 0], dtype=jnp.complex128)  # |00⟩
        result = fidelity(sv0, sv1)
        expected = qml.math.fidelity_statevector(np.array(sv0), np.array(sv1))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_trace_distance_2qubit(self):
        """Trace distance of 2-qubit density matrices should match PennyLane."""
        bell = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])
        rho0 = jnp.outer(bell, jnp.conj(bell))
        rho1 = jnp.eye(4, dtype=jnp.complex128) / 4  # maximally mixed
        result = trace_distance(rho0, rho1)
        expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "sv0,sv1,expected_val",
        [
            (
                jnp.array([1.0, 0.0]),
                jnp.array([1.0, 0.0]),
                0.0,
            ),
            (
                jnp.array([1.0, 0.0]),
                jnp.array([-1.0, 0.0]),
                jnp.pi,
            ),
            (
                jnp.array([1.0, 0.0]),
                jnp.array([1j, 0.0]),
                jnp.pi / 2,
            ),
            (
                jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j]),
                jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j]),
                None,  # just check against vdot formula
            ),
        ],
        ids=["identical", "global_phase", "half_pi", "arbitrary"],
    )
    def test_phase_difference(self, sv0, sv1, expected_val):
        """Phase difference should match the vdot formula."""
        result = phase_difference(sv0, sv1)
        expected = jnp.angle(jnp.vdot(sv0, sv1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_phase_difference_batched(self):
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
            expected_i = jnp.angle(inner)
            assert jnp.allclose(result[i], expected_i, atol=1e-10)

    @pytest.mark.unittest
    def test_phase_difference_2qubit(self):
        """Phase difference of 2-qubit states should match the vdot formula."""
        sv0 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])  # Bell |Φ+⟩
        sv1 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1j / jnp.sqrt(2)])  # phase on |11⟩
        result = phase_difference(sv0, sv1)
        expected = jnp.angle(jnp.vdot(sv0, sv1))
        assert jnp.allclose(result, expected, atol=1e-10)


class TestPulse:
    @pytest.mark.unittest
    def test_collect_pulse_events_rx(self):
        """collect_pulse_events for RX returns a single physical pulse event."""
        from qml_essentials.drawing import collect_pulse_events

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")
            events = collect_pulse_events("RX", jnp.pi / 4, wires=0)
            assert len(events) == 1
            ev = events[0]
            assert ev.gate == "RX"
            assert ev.wires == [0]
            assert ev.envelope_fn is not None
            assert ev.duration > 0
            assert np.isclose(ev.w, jnp.pi / 4)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_collect_pulse_events_rz(self):
        """collect_pulse_events for RZ returns a virtual-Z event (no envelope_fn)."""
        from qml_essentials.drawing import collect_pulse_events

        events = collect_pulse_events("RZ", jnp.pi / 3, wires=1)
        assert len(events) == 1
        assert events[0].gate == "RZ"
        assert events[0].envelope_fn is None

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "gate_name,params,wires,expected_gates,parent",
        [
            ("H", 0.0, 0, ["RZ", "RY"], "H"),
            ("CX", 0.0, [0, 1], ["RZ", "RY", "CZ", "RZ", "RY"], None),
            ("Rot", [jnp.pi / 4, jnp.pi / 2, jnp.pi / 3], 0, ["RZ", "RY", "RZ"], "Rot"),
        ],
        ids=["H", "CX", "Rot"],
    )
    def test_collect_pulse_events_decomposes(
        self, gate_name, params, wires, expected_gates, parent
    ):
        """Composite gates decompose into expected leaf pulse events."""
        from qml_essentials.drawing import collect_pulse_events

        events = collect_pulse_events(gate_name, params, wires=wires)
        gate_names = [ev.gate for ev in events]
        assert gate_names == expected_gates
        if parent is not None:
            for ev in events:
                assert ev.parent == parent

    @pytest.mark.unittest
    def test_collect_pulse_events_invalid_gate(self):
        """Unknown gate name raises ValueError."""
        from qml_essentials.drawing import collect_pulse_events

        with pytest.raises(ValueError, match="Unknown pulse gate"):
            collect_pulse_events("INVALID", 0.0, wires=0)

    @pytest.mark.unittest
    def test_draw_pulse_schedule_returns_figure(self):
        """draw_pulse_schedule returns a matplotlib (fig, axes) tuple."""
        from qml_essentials.drawing import collect_pulse_events, draw_pulse_schedule

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")
            events = collect_pulse_events("RX", jnp.pi / 2, wires=0)
            fig, axes = draw_pulse_schedule(events, n_qubits=1)

            import matplotlib.pyplot as plt

            assert fig is not None
            assert len(axes) == 1
            plt.close(fig)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_draw_pulse_schedule_multi_qubit(self):
        """Pulse schedule with CX renders subplots for both qubits."""
        from qml_essentials.drawing import collect_pulse_events, draw_pulse_schedule

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")
            events = collect_pulse_events("CX", 0.0, wires=[0, 1])
            fig, axes = draw_pulse_schedule(events, n_qubits=2)

            import matplotlib.pyplot as plt

            assert len(axes) == 2
            plt.close(fig)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_draw_pulse_schedule_show_carrier(self):
        """show_carrier=True should not raise."""
        from qml_essentials.drawing import collect_pulse_events, draw_pulse_schedule

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")
            events = collect_pulse_events("RY", jnp.pi / 4, wires=0)
            fig, axes = draw_pulse_schedule(events, n_qubits=1, show_carrier=True)

            import matplotlib.pyplot as plt

            plt.close(fig)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    @pytest.mark.parametrize("envelope", PulseEnvelope.available())
    def test_draw_pulse_schedule_all_envelopes(self, envelope):
        """Pulse schedule renders without error for every envelope."""
        from qml_essentials.drawing import collect_pulse_events, draw_pulse_schedule

        if envelope == "general":
            return

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope(envelope)
            events = collect_pulse_events("RX", jnp.pi / 3, wires=0)
            fig, axes = draw_pulse_schedule(events, n_qubits=1)

            import matplotlib.pyplot as plt

            plt.close(fig)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_pulse_envelope_available(self):
        """All expected envelope names are registered."""
        names = PulseEnvelope.available()
        for expected in ["gaussian", "square", "cosine", "drag", "sech"]:
            assert (
                expected in names
            ), f"'{expected}' missing from PulseEnvelope.available()"

    @pytest.mark.unittest
    def test_pulse_envelope_get_valid(self):
        """get() returns metadata dict with required keys."""
        for name in PulseEnvelope.available():
            info = PulseEnvelope.get(name)
            assert "fn" in info
            assert "n_envelope_params" in info
            assert "defaults" in info
            if name != "general":
                assert callable(info["fn"])
                for gate in ["RX", "RY"]:
                    assert (
                        gate in info["defaults"]
                    ), f"Missing default for gate '{gate}' in envelope '{name}'"
            else:
                assert info["fn"] is None
                for gate in ["RZ", "CZ"]:
                    assert (
                        gate in info["defaults"]
                    ), f"Missing default for gate '{gate}' in envelope '{name}'"

    @pytest.mark.unittest
    def test_pulse_envelope_get_invalid(self):
        """get() raises ValueError for unknown envelope names."""
        with pytest.raises(ValueError, match="Unknown pulse envelope"):
            PulseEnvelope.get("nonexistent_envelope")

    @pytest.mark.unittest
    def test_pulse_envelope_functions_callable(self):
        """Each envelope function runs without error for typical inputs."""
        t = 0.5
        t_c = 0.25
        test_args = {
            "gaussian": jnp.array([1.0, 0.5]),
            "square": jnp.array([1.0, 0.5]),
            "cosine": jnp.array([1.0, 0.5]),
            "drag": jnp.array([1.0, 0.5, 0.1]),
            "sech": jnp.array([1.0, 0.5]),
        }
        for name, p in test_args.items():
            fn = PulseEnvelope.get(name)["fn"]
            result = fn(p, t, t_c)
            assert jnp.isfinite(result), f"Envelope '{name}' returned non-finite value"

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "envelope_fn,name",
        [
            (PulseEnvelope.gaussian, "gaussian"),
            (PulseEnvelope.sech, "sech"),
        ],
        ids=["gaussian", "sech"],
    )
    def test_pulse_envelope_peak(self, envelope_fn, name):
        """Envelope peaks at t == t_c with value A."""
        A, sigma = 2.0 if name == "gaussian" else 3.0, 1.0
        p = jnp.array([A, sigma])
        t_c = 0.5
        val_at_center = envelope_fn(p, t_c, t_c)
        val_off_center = envelope_fn(p, t_c + 2 * sigma, t_c)
        assert jnp.isclose(val_at_center, A, atol=1e-10)
        assert val_off_center < val_at_center

    @pytest.mark.unittest
    def test_pulse_envelope_drag_reduces_to_gaussian(self):
        """DRAG with beta=0 reduces to Gaussian."""
        A, sigma = 2.0, 1.0
        p_gauss = jnp.array([A, sigma])
        p_drag = jnp.array([A, 0.0, sigma])
        t, t_c = 0.3, 0.5
        g = PulseEnvelope.gaussian(p_gauss, t, t_c)
        d = PulseEnvelope.drag(p_drag, t, t_c)
        assert jnp.isclose(g, d, atol=1e-10)

    @pytest.mark.unittest
    def test_build_coeff_fns_unique_code(self):
        """build_coeff_fns for different envelopes produces different outputs."""
        omega_c = PulseGates.omega_c
        omega_q = PulseGates.omega_q
        rxx_g, rxy_g, ryx_g, ryy_g = PulseEnvelope.build_coeff_fns(
            PulseEnvelope.gaussian, omega_c, omega_q
        )
        rxx_d, rxy_d, ryx_d, ryy_d = PulseEnvelope.build_coeff_fns(
            PulseEnvelope.drag, omega_c, omega_q
        )
        p_gauss = jnp.array([1.0, 1.0, 1.0])  # [A, sigma, w]
        p_drag = jnp.array([1.0, 0.5, 1.0, 1.0])  # [A, beta, sigma, w]
        t = 0.123
        # Different envelopes → different coefficient values
        assert not jnp.allclose(rxx_g(p_gauss, t), rxx_d(p_drag, t))
        assert not jnp.allclose(ryy_g(p_gauss, t), ryy_d(p_drag, t))
        # RX and RY dominant components differ (different carrier phases)
        assert not jnp.allclose(rxx_g(p_gauss, t), ryy_g(p_gauss, t))

    @pytest.mark.unittest
    def test_pulse_information_set_envelope(self):
        """set_envelope updates PulseInformation and PulseGates state."""
        original_envelope = PulseInformation.get_envelope()

        try:
            PulseInformation.set_envelope("drag")
            assert PulseInformation.get_envelope() == "drag"
            assert PulseGates._active_envelope == "drag"
            # DRAG has 3 envelope params → RX defaults should have 4 elements
            assert len(PulseInformation.RX.params) == 4

            PulseInformation.set_envelope("gaussian")
            assert PulseInformation.get_envelope() == "gaussian"
            assert PulseGates._active_envelope == "gaussian"
            # Gaussian has 2 envelope params → RX defaults should have 3 elements
            assert len(PulseInformation.RX.params) == 3
        finally:
            PulseInformation.set_envelope(original_envelope)

    @pytest.mark.unittest
    def test_pulse_information_set_envelope_invalid(self):
        """set_envelope raises ValueError for unknown names."""
        with pytest.raises(ValueError, match="Unknown pulse envelope"):
            PulseInformation.set_envelope("banana")

    @pytest.mark.unittest
    def test_pulse_information_param_counts_per_envelope(self):
        """Parameter counts for composite gates update when envelope changes."""
        original = PulseInformation.get_envelope()
        try:
            for name in PulseEnvelope.available():
                if name == "general":
                    continue
                PulseInformation.set_envelope(name)
                info = PulseEnvelope.get(name)
                # RX params = n_envelope_params + 1 (time)
                expected_rx = info["n_envelope_params"] + 1
                assert len(PulseInformation.RX.params) == expected_rx, (
                    f"RX param count wrong for envelope '{name}': "
                    f"expected {expected_rx}, got {len(PulseInformation.RX.params)}"
                )
                # H = RZ + RY → sum of their param counts
                expected_h = len(PulseInformation.RZ.params) + len(
                    PulseInformation.RY.params
                )
                assert (
                    PulseInformation.H.size == expected_h
                ), f"H param count wrong for envelope '{name}'"
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_pulse_rx_gaussian_fidelity(self):
        """PulseGates.RX with gaussian envelope produces correct state."""
        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")
            w = jnp.pi / 4

            def pulse_circuit(w, pp):
                PulseGates.RX(w, wires=0, pulse_params=pp)

            def target_circuit(w):
                from qml_essentials.operations import RX as OpRX

                OpRX(w, wires=0)

            pulse_script = Script(pulse_circuit, n_qubits=1)
            target_script = Script(target_circuit, n_qubits=1)

            state_pulse = pulse_script.execute(
                type="state", args=(w, PulseInformation.RX.params)
            )
            state_target = target_script.execute(type="state", args=(w,))

            f = jnp.abs(jnp.vdot(state_target, state_pulse)) ** 2
            assert f <= 1.0 + 1e-6
            assert np.isclose(f, 1.0, atol=1e-2), f"Fidelity too low: {f}"
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    @pytest.mark.parametrize("envelope", PulseEnvelope.available())
    def test_pulse_rz_all_envelopes(self, envelope):
        """PulseGates.RZ produces correct state for every registered envelope.

        RZ is a virtual-Z gate (no physical pulse), so it should work
        identically regardless of envelope.
        """
        if envelope == "general":
            return

        original = PulseInformation.get_envelope()

        try:
            PulseInformation.set_envelope(envelope)
            w = jnp.pi / 3

            def pulse_circuit(w, pp):
                PulseGates.RZ(w, wires=0, pulse_params=pp)

            def target_circuit(w):
                from qml_essentials.operations import RZ as OpRZ

                OpRZ(w, wires=0)

            pulse_script = Script(pulse_circuit, n_qubits=1)
            target_script = Script(target_circuit, n_qubits=1)

            state_pulse = pulse_script.execute(
                type="state", args=(w, PulseInformation.RZ.params)
            )
            state_target = target_script.execute(type="state", args=(w,))

            f = jnp.abs(jnp.vdot(state_target, state_pulse)) ** 2
            assert np.isclose(
                f, 1.0, atol=1e-2
            ), f"RZ fidelity too low for envelope '{envelope}': {f}"
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_set_envelope_updates_coeff_fns(self):
        """Switching envelope actually changes the coefficient function outputs."""
        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")
            sx_gaussian = PulseGates._coeff_Sx

            PulseInformation.set_envelope("sech")
            sx_sech = PulseGates._coeff_Sx

            # The coefficient functions must be different objects
            assert sx_gaussian is not sx_sech
            # And produce different numerical results
            p = jnp.array([1.0, 1.0, 1.0])  # [A, sigma, w]
            t = 0.3
            assert not jnp.allclose(sx_gaussian(p, t), sx_sech(p, t))
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_set_envelope_roundtrip(self):
        """Switching away and back restores the same parameter values."""
        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")
            rx_params_before = PulseInformation.RX.params.copy()
            rx_len_before = len(rx_params_before)

            PulseInformation.set_envelope("drag")
            # DRAG has different param count, so length must differ
            assert len(PulseInformation.RX.params) != rx_len_before

            PulseInformation.set_envelope("gaussian")
            assert jnp.allclose(PulseInformation.RX.params, rx_params_before)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_rwa_toggle_default_off(self):
        """Default RWA flag is False."""
        assert PulseInformation.get_rwa() is False
        assert PulseGates._active_rwa is False

    @pytest.mark.unittest
    def test_rwa_toggle_roundtrip(self):
        """set_rwa flips the flag, rebuilds coeffs, and restores cleanly."""
        original_env = PulseInformation.get_envelope()
        original_rwa = PulseInformation.get_rwa()
        try:
            PulseInformation.set_envelope("gaussian")
            PulseInformation.set_rwa(True)
            assert PulseInformation.get_rwa() is True
            assert PulseGates._active_rwa is True

            PulseInformation.set_rwa(False)
            assert PulseInformation.get_rwa() is False
            assert PulseGates._active_rwa is False
        finally:
            PulseInformation.set_envelope(original_env)
            PulseInformation.set_rwa(original_rwa)

    @pytest.mark.unittest
    def test_rwa_coeffs_drop_fast_oscillations(self):
        """RWA coeffs are envelope-only (no carrier); exact form has both.

        Direct check that ``rwa=True`` removes the fast factors

        - Exact: ``c_X(t) = env·cos(ω_c t)·cos(ω_q t)·w``
          → vanishes whenever ``cos(ω_c t) = 0``.
        - RWA:   ``c_X(t) = 0.5·env·w``
          → never vanishes for non-zero ``env`` and ``w``.
        """
        omega_c = PulseGates.omega_c
        omega_q = PulseGates.omega_q

        rxx_exact, rxy_exact, ryx_exact, ryy_exact = PulseEnvelope.build_coeff_fns(
            PulseEnvelope.gaussian, omega_c, omega_q, rwa=False
        )
        rxx_rwa, rxy_rwa, ryx_rwa, ryy_rwa = PulseEnvelope.build_coeff_fns(
            PulseEnvelope.gaussian, omega_c, omega_q, rwa=True
        )

        p = jnp.array([1.0, 0.5, 1.0])  # [A, sigma, w]
        # Time at which the carrier vanishes (cos(omega_c t) = 0).
        t_zero = jnp.pi / (2 * omega_c)
        # Reference time at which envelope and carrier are positive.
        t_ref = 0.0

        # Exact form respects the carrier zero crossing.
        assert jnp.isclose(rxx_exact(p, t_zero), 0.0, atol=1e-10)
        assert jnp.isclose(rxy_exact(p, t_zero), 0.0, atol=1e-10)
        # RWA form does not (envelope is finite at t_zero).
        assert jnp.abs(rxx_rwa(p, t_zero)) > 1e-3
        # And RWA equals the closed-form ``0.5 * env(p, t, t/2) * w``.
        env_val = PulseEnvelope.gaussian(p, t_ref, t_ref / 2)
        assert jnp.isclose(rxx_rwa(p, t_ref), 0.5 * env_val * p[-1], atol=1e-10)
        assert jnp.isclose(ryy_rwa(p, t_ref), 0.5 * env_val * p[-1], atol=1e-10)
        # Off-diagonal RWA components are identically zero.
        assert jnp.isclose(rxy_rwa(p, 0.123), 0.0, atol=1e-12)
        assert jnp.isclose(ryx_rwa(p, 0.123), 0.0, atol=1e-12)

    @pytest.mark.unittest
    def test_rwa_unitary_matches_envelope_area(self):
        """RWA RX rotation angle equals ``w × ∫env dt`` (closed-form).

        In RWA mode ``H_I(t) = (env(t)/2)·w·X``.  Integrating gives
        ``U = exp(-i (w·Area/2) X)`` where ``Area = ∫_0^t_g env dt``.
        The numerical pulse must agree with the analytic ``OpRX`` at
        the *effective* angle ``theta_eff = w · Area``.  Computed
        Area via a fine trapezoidal quadrature so the test is
        independent of any per-envelope calibration.
        """
        original_env = PulseInformation.get_envelope()
        original_rwa = PulseInformation.get_rwa()
        try:
            PulseInformation.set_envelope("gaussian")
            PulseInformation.set_rwa(True)

            A, sigma, t_g = 0.5, 0.4, 1.5
            w = 1.0
            pp = jnp.array([A, sigma, t_g])

            # Closed-form area of env(τ) = A·exp(-(τ/2)^2/(2 sigma^2))
            # over [0, t_g] via dense trapezoid (no JAX needed).
            ts = jnp.linspace(0.0, t_g, 2048)
            env_vals = jax.vmap(
                lambda tau: PulseEnvelope.gaussian(jnp.array([A, sigma]), tau, tau / 2)
            )(ts)
            area = jnp.trapezoid(env_vals, ts)
            theta_eff = float(w * area)

            def pulse_circuit(w, pp):
                PulseGates.RX(w, wires=0, pulse_params=pp)

            def target_circuit(theta):
                from qml_essentials.operations import RX as OpRX

                OpRX(theta, wires=0)

            pulse_script = Script(pulse_circuit, n_qubits=1)
            target_script = Script(target_circuit, n_qubits=1)

            state_pulse = pulse_script.execute(type="state", args=(w, pp))
            state_target = target_script.execute(type="state", args=(theta_eff,))

            f = jnp.abs(jnp.vdot(state_target, state_pulse)) ** 2
            assert f <= 1.0 + 1e-6
            # RWA mode is tight: no fast oscillations to integrate.
            assert np.isclose(
                f, 1.0, atol=5e-3
            ), f"RWA RX fidelity (theta_eff={theta_eff:.4f}) too low: {f}"
        finally:
            PulseInformation.set_envelope(original_env)
            PulseInformation.set_rwa(original_rwa)

    # ---------------------------------------------------------------------------
    # Solver / frame regression tests (Magnus integrator + drive-frame mode)
    # ---------------------------------------------------------------------------

    @pytest.mark.unittest
    def test_solver_default_dopri8(self):
        """Default solver is dopri8 (adaptive RK)."""
        from qml_essentials.yaqsi import Yaqsi

        assert Yaqsi._solver_defaults["solver"] == "dopri8"

    @pytest.mark.unittest
    def test_solver_invalid_name_raises(self):
        """Unknown solver names raise ValueError."""
        from qml_essentials.yaqsi import Yaqsi

        with pytest.raises(ValueError):
            Yaqsi.set_solver_defaults(solver="foobar")

    @pytest.mark.unittest
    def test_magnus_matches_dopri8_rx(self):
        """magnus2 / magnus4 reproduce dopri8 unitaries to high accuracy."""
        from qml_essentials.yaqsi import Yaqsi
        import qml_essentials.operations as op_mod

        original_env = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("drag")
            flat = PulseInformation.RX.params
            H_X = op_mod.Hermitian(PulseGates.X, wires=0, record=False)
            H_Y = op_mod.Hermitian(PulseGates.Y, wires=0, record=False)
            H_eff = PulseGates._coeff_RX_X * H_X + PulseGates._coeff_RX_Y * H_Y

            w = float(jnp.pi / 2)
            t_g = float(flat[-1])
            args = [jnp.array([*flat[:-1], w])] * 2

            U_ref = Yaqsi.evolve(H_eff, name="RX", atol=1e-12, rtol=1e-12)(
                args, t_g
            ).matrix
            U_m2 = Yaqsi.evolve(H_eff, name="RX", solver="magnus2", magnus_steps=2048)(
                args, t_g
            ).matrix
            U_m4 = Yaqsi.evolve(H_eff, name="RX", solver="magnus4", magnus_steps=512)(
                args, t_g
            ).matrix

            assert float(jnp.linalg.norm(U_m2 - U_ref)) < 1e-3
            assert float(jnp.linalg.norm(U_m4 - U_ref)) < 1e-5

            Id = jnp.eye(2, dtype=U_m4.dtype)
            assert float(jnp.linalg.norm(U_m4.conj().T @ U_m4 - Id)) < 1e-10
            assert float(jnp.linalg.norm(U_m2.conj().T @ U_m2 - Id)) < 1e-10
        finally:
            PulseInformation.set_envelope(original_env)

    @pytest.mark.unittest
    def test_magnus4_fourth_order_convergence(self):
        """magnus4 error scales as h^4 (≈16× drop per N doubling)."""
        from qml_essentials.yaqsi import Yaqsi
        import qml_essentials.operations as op_mod

        original_env = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("drag")
            flat = PulseInformation.RX.params
            H_X = op_mod.Hermitian(PulseGates.X, wires=0, record=False)
            H_Y = op_mod.Hermitian(PulseGates.Y, wires=0, record=False)
            H_eff = PulseGates._coeff_RX_X * H_X + PulseGates._coeff_RX_Y * H_Y
            w = float(jnp.pi / 2)
            t_g = float(flat[-1])
            args = [jnp.array([*flat[:-1], w])] * 2
            U_ref = Yaqsi.evolve(H_eff, name="RX", atol=1e-12, rtol=1e-12)(
                args, t_g
            ).matrix

            errs = []
            for N in (256, 512, 1024):
                U_m = Yaqsi.evolve(
                    H_eff,
                    name="RX",
                    solver="magnus4",
                    magnus_steps=N,
                )(args, t_g).matrix
                errs.append(float(jnp.linalg.norm(U_m - U_ref)))

            assert (
                errs[0] / errs[1] > 8.0
            ), f"magnus4 not 4th-order: {errs[0]:.3e} / {errs[1]:.3e}"
            assert errs[1] / errs[2] > 8.0
        finally:
            PulseInformation.set_envelope(original_env)

    @pytest.mark.unittest
    def test_drive_frame_default_lab(self):
        """Default coefficient frame is 'lab'."""
        assert PulseInformation.get_frame() == "lab"
        assert PulseGates._active_frame == "lab"

    @pytest.mark.unittest
    def test_drive_frame_invalid_raises(self):
        """Unknown frame names raise ValueError."""
        from qml_essentials.pulses import PulseEnvelope

        with pytest.raises(ValueError):
            PulseEnvelope.build_coeff_fns(PulseEnvelope.drag, 1.0, 1.0, frame="foobar")
        with pytest.raises(ValueError):
            PulseInformation.set_frame("foobar")

    @pytest.mark.unittest
    def test_drive_frame_equivalent_to_lab(self):
        """drive-frame coefficients equal lab-frame to machine precision."""
        from qml_essentials.pulses import PulseEnvelope

        for omega_c, omega_q in [(1.234, 1.234), (1.5, 1.0), (3.0, 7.0)]:
            lab = PulseEnvelope.build_coeff_fns(
                PulseEnvelope.drag, omega_c, omega_q, frame="lab"
            )
            drv = PulseEnvelope.build_coeff_fns(
                PulseEnvelope.drag, omega_c, omega_q, frame="drive"
            )
            p = jnp.array([0.5, 0.3, 5.0, jnp.pi / 2])
            ts = jnp.linspace(0.0, 4.0, 50)
            for fl, fd in zip(lab, drv):
                vals_lab = jnp.array([fl(p, float(t)) for t in ts])
                vals_drv = jnp.array([fd(p, float(t)) for t in ts])
                assert float(jnp.max(jnp.abs(vals_lab - vals_drv))) < 1e-12

    @pytest.mark.unittest
    def test_drive_frame_roundtrip(self):
        """set_frame switches PulseGates._active_frame and restores."""
        original_env = PulseInformation.get_envelope()
        original_frame = PulseInformation.get_frame()
        try:
            PulseInformation.set_frame("drive")
            assert PulseInformation.get_frame() == "drive"
            assert PulseGates._active_frame == "drive"
            PulseInformation.set_frame("lab")
            assert PulseInformation.get_frame() == "lab"
            assert PulseGates._active_frame == "lab"
        finally:
            PulseInformation.set_envelope(original_env, frame=original_frame)

    # ---------------------------------------------------------------------------
    # pulse_recording / Script.pulse_events tests
    # ---------------------------------------------------------------------------

    @pytest.mark.unittest
    def test_pulse_recording_rx(self):
        """pulse_recording captures a PulseEvent for a single RX gate."""
        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")

            def circuit(w, pp):
                PulseGates.RX(w, wires=0, pulse_params=pp)

            script = Script(circuit, n_qubits=1)
            events = script.pulse_events(jnp.pi / 4, PulseInformation.RX.params)

            assert len(events) == 1
            ev = events[0]
            assert ev.gate == "RX"
            assert ev.wires == [0]
            assert ev.envelope_fn is not None
            assert np.isclose(ev.w, jnp.pi / 4)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "gate_name,gate_fn,wires,n_qubits,expected_gates",
        [
            ("H", lambda: PulseGates.H(wires=0), [0], 1, ["RZ", "RY"]),
            (
                "CX",
                lambda: PulseGates.CX(wires=[0, 1]),
                [0, 1],
                2,
                ["RZ", "RY", "CZ", "RZ", "RY"],
            ),
        ],
        ids=["H", "CX"],
    )
    def test_pulse_recording_composite(
        self, gate_name, gate_fn, wires, n_qubits, expected_gates
    ):
        """pulse_recording captures correct leaf events from composite gates."""
        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")

            def circuit():
                gate_fn()

            script = Script(circuit, n_qubits=n_qubits)
            events = script.pulse_events()

            gate_names = [ev.gate for ev in events]
            assert gate_names == expected_gates
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_pulse_recording_context_direct(self):
        """pulse_recording context manager works without Script."""
        from qml_essentials.tape import pulse_recording, recording

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")

            with pulse_recording() as events:
                with recording():
                    PulseGates.RY(jnp.pi / 2, wires=0)

            assert len(events) == 1
            assert events[0].gate == "RY"
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_pulse_recording_no_events_outside_context(self):
        """No events captured when pulse_recording is not active."""
        from qml_essentials.tape import active_pulse_tape, recording

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")

            assert active_pulse_tape() is None

            with recording():
                PulseGates.RX(jnp.pi, wires=0)
            # No crash, no events captured
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_script_draw_pulse_mode(self):
        """Script.draw(figure='pulse') returns a matplotlib figure."""
        import matplotlib.pyplot as plt

        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")

            def circuit(w, pp):
                PulseGates.RX(w, wires=0, pulse_params=pp)

            script = Script(circuit, n_qubits=1)
            fig, axes = script.draw(
                figure="pulse",
                args=(jnp.pi / 2, PulseInformation.RX.params),
            )

            assert fig is not None
            assert len(axes) == 1
            plt.close(fig)
        finally:
            PulseInformation.set_envelope(original)

    @pytest.mark.unittest
    def test_pulse_recording_multi_gate_sequence(self):
        """pulse_recording captures events from a multi-gate sequence."""
        original = PulseInformation.get_envelope()
        try:
            PulseInformation.set_envelope("gaussian")

            def circuit(w1, w2):
                PulseGates.RX(w1, wires=0)
                PulseGates.RY(w2, wires=1)
                PulseGates.CZ(wires=[0, 1])

            script = Script(circuit, n_qubits=2)
            events = script.pulse_events(jnp.pi / 4, jnp.pi / 3)

            gate_names = [ev.gate for ev in events]
            assert gate_names == ["RX", "RY", "CZ"]
            assert events[0].wires == [0]
            assert events[1].wires == [1]
            assert events[2].wires == [0, 1]
        finally:
            PulseInformation.set_envelope(original)
