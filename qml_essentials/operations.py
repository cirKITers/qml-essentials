import threading
from typing import List, Optional, Union

import jax
import jax.numpy as jnp

# Enable 64-bit precision (important for quantum simulation accuracy)
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Global tape (thread-local so that concurrent scripts don't interfere)
# ---------------------------------------------------------------------------
_tape_local = threading.local()


def _active_tape() -> Optional[List["Operation"]]:
    """Return the currently recording tape, or None."""
    return getattr(_tape_local, "tape", None)


def _set_tape(tape: Optional[List["Operation"]]):
    _tape_local.tape = tape


# ---------------------------------------------------------------------------
# Core gate matrices (constant, defined once)
# ---------------------------------------------------------------------------
I = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
H = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
CNOT = jnp.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=jnp.complex128,
)


# ===================================================================
# Operation base class
# ===================================================================
class Operation:
    """
    This forms the basis for any quantum operation.
    Further gates should inherit from this class to realize
    more specific operations.
    Generally, operations should be applied by instantiation
    and appending to a QuantumScript.
    This script should then be executed once the circuit is called.

    An Operation can also serve as an *observable* (its matrix is used
    to compute expectation values).
    """

    # Subclasses should set this to the gate's unitary / matrix
    _matrix: jnp.ndarray = None

    def __init__(
        self,
        wires: Union[int, List[int]] = 0,
        matrix: Optional[jnp.ndarray] = None,
    ):
        self.wires = wires if isinstance(wires, list) else [wires]
        if matrix is not None:
            self._matrix = matrix

        # If a tape is currently recording, append ourselves
        tape = _active_tape()
        if tape is not None:
            tape.append(self)

    @property
    def matrix(self) -> jnp.ndarray:
        """Return the base matrix of this operation (before lifting)."""
        if self._matrix is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not define a matrix."
            )
        return self._matrix

    @property
    def wires(self) -> List[int]:
        return self._wires

    @wires.setter
    def wires(self, wires: Union[int, List[int]]):
        if isinstance(wires, int):
            self._wires = [wires]
        else:
            self._wires = wires

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """
        Apply this gate to a **statevector** via tensor contraction.

        The statevector (shape ``(2**n,)``) is reshaped into a rank-*n* tensor
        of shape ``(2,)*n``.  The gate (shape ``(2**k, 2**k)``) is reshaped to
        ``(2,)*2k`` and contracted against the *k* target wire axes.

        Memory:  O(2**n).  Supports arbitrary k.  Fully differentiable.
        """
        k = len(self.wires)
        gate_tensor = self.matrix.reshape((2,) * 2 * k)
        psi = state.reshape((2,) * n_qubits)

        psi_out = jnp.tensordot(
            gate_tensor, psi, axes=(list(range(k, 2 * k)), self.wires)
        )

        # Restore axis order: tensordot places the k output axes first,
        # followed by the n-k untouched axes in their original relative order.
        remaining = [q for q in range(n_qubits) if q not in self.wires]
        dest = list(self.wires) + remaining
        perm = [0] * n_qubits
        for src_pos, dst_pos in enumerate(dest):
            perm[dst_pos] = src_pos
        psi_out = jnp.transpose(psi_out, perm)

        return psi_out.reshape(2**n_qubits)

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """
        Apply this gate to a **density matrix** via ρ → UρU†.

        The density matrix (shape ``(2**n, 2**n)``) is treated as a rank-*2n*
        tensor with *n* "ket" axes (0..n-1) and *n* "bra" axes (n..2n-1).
        U acts on the ket half; U* acts on the bra half.  Both contractions
        reuse the same axis-permutation logic as ``apply_to_state``, keeping
        the operation allocation-free w.r.t. building full unitaries.

        This is the correct building block for future noise channels, where a
        mixed state cannot be represented as a pure statevector.
        """
        k = len(self.wires)
        U = self.matrix.reshape((2,) * 2 * k)  # (out)*k + (in)*k
        U_conj = jnp.conj(U)

        # Represent ρ as a (2,)*2n tensor: axes 0..n-1 = ket, n..2n-1 = bra
        rho_t = rho.reshape((2,) * 2 * n_qubits)

        # Helper: apply a (2,)*2k gate tensor to k chosen axes of rho_t,
        # then restore the correct axis order for a 2n-rank tensor.
        def _contract_and_restore(tensor, gate, target_axes):
            """
            Contract gate (last k indices) against `target_axes` of `tensor`,
            placing k new output axes at the *front* of the result, then
            restore them to `target_axes` positions.

            tensor shape: (2,)*2n
            target_axes:  k ints, each in [0, 2n)
            """
            total = 2 * n_qubits
            out = jnp.tensordot(gate, tensor, axes=(list(range(k, 2 * k)), target_axes))
            # tensordot output: k new axes first, then (2n-k) remaining axes
            # in the same relative order they had in `tensor` minus target_axes.
            remaining = [ax for ax in range(total) if ax not in target_axes]
            # dest[i] = where the i-th output axis should go in the final tensor
            dest = list(target_axes) + remaining
            # Build inverse permutation
            perm = [0] * total
            for src_pos, dst_pos in enumerate(dest):
                perm[dst_pos] = src_pos
            return jnp.transpose(out, perm)

        # Apply U  to ket axes (self.wires)
        rho_t = _contract_and_restore(rho_t, U, self.wires)
        # Apply U† to bra axes (self.wires + n_qubits)
        bra_wires = [w + n_qubits for w in self.wires]
        rho_t = _contract_and_restore(rho_t, U_conj, bra_wires)

        return rho_t.reshape(2**n_qubits, 2**n_qubits)


# ===================================================================
# Concrete gates / observables
# ===================================================================
class Hermitian(Operation):
    """
    A generic Hermitian observable / gate defined by an arbitrary matrix.

    Usage:
        obs = Hermitian(matrix=my_matrix, wires=0)
    """

    def __init__(self, matrix: jnp.ndarray, wires: Union[int, List[int]] = 0):
        super().__init__(wires=wires, matrix=jnp.asarray(matrix, dtype=jnp.complex128))


class PauliX(Operation):
    """Pauli-X gate / observable."""

    _matrix = X

    def __init__(self, wires: Union[int, List[int]] = 0):
        super().__init__(wires=wires)


class PauliY(Operation):
    """Pauli-Y gate / observable."""

    _matrix = Y

    def __init__(self, wires: Union[int, List[int]] = 0):
        super().__init__(wires=wires)


class PauliZ(Operation):
    """Pauli-Z gate / observable."""

    _matrix = Z

    def __init__(self, wires: Union[int, List[int]] = 0):
        super().__init__(wires=wires)


class H(Operation):
    """Hadamard gate."""

    _matrix = H

    def __init__(self, wires: Union[int, List[int]] = 0):
        super().__init__(wires=wires)


class CX(Operation):
    """Controlled-X (CNOT) gate.  wires=[control, target]."""

    _matrix = CNOT

    def __init__(self, wires: List[int] = [0, 1]):
        super().__init__(wires=wires)


class CCX(Operation):
    """Toffoli (CCX) gate.  wires=[control0, control1, target].

    The 3-qubit Toffoli gate exercises the arbitrary-k-qubit path in
    ``apply_to_state`` and cannot be expressed as a pair of 2-qubit gates
    without ancilla, making it a good stress-test for the simulator.
    """

    _matrix = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=jnp.complex128,
    )

    def __init__(self, wires: List[int] = [0, 1, 2]):
        super().__init__(wires=wires)


# ===================================================================
# Kraus channel base class
# ===================================================================
class KrausChannel(Operation):
    """
    Base class for noise channels defined by a set of Kraus operators.

    A Kraus channel Φ(ρ) = Σ_k K_k ρ K_k† is the most general physical
    operation on a quantum state. For pure unitary gates Σ_k K_k†K_k = I
    with a single K_0 = U; for noisy channels there are multiple operators.

    Subclasses must implement ``kraus_matrices()`` returning a list of
    JAX arrays.  ``apply_to_state`` is intentionally left unimplemented:
    Kraus channels require a density-matrix representation and cannot be
    applied to a pure statevector in general.
    """

    def kraus_matrices(self) -> List[jnp.ndarray]:
        raise NotImplementedError

    @property
    def matrix(self) -> jnp.ndarray:
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and has no single "
            "unitary matrix. Use apply_to_density() instead."
        )

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and cannot be "
            "applied to a pure statevector. Use execute(type='density') instead."
        )

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply Φ(ρ) = Σ_k K_k ρ K_k† using the same tensor-contraction
        engine as Operation.apply_to_density, but summing over all Kraus ops."""
        k = len(self.wires)
        total = 2 * n_qubits
        rho_out = jnp.zeros_like(rho)

        def _contract_and_restore(tensor, gate, target_axes):
            out = jnp.tensordot(gate, tensor, axes=(list(range(k, 2 * k)), target_axes))
            remaining = [ax for ax in range(total) if ax not in target_axes]
            dest = list(target_axes) + remaining
            perm = [0] * total
            for src_pos, dst_pos in enumerate(dest):
                perm[dst_pos] = src_pos
            return jnp.transpose(out, perm)

        for K in self.kraus_matrices():
            K_t = K.reshape((2,) * 2 * k)
            K_conj_t = jnp.conj(K_t)
            rho_t = rho.reshape((2,) * total)
            # apply K to ket axes
            rho_t = _contract_and_restore(rho_t, K_t, self.wires)
            # apply K† to bra axes
            bra_wires = [w + n_qubits for w in self.wires]
            rho_t = _contract_and_restore(rho_t, K_conj_t, bra_wires)
            rho_out = rho_out + rho_t.reshape(2**n_qubits, 2**n_qubits)

        return rho_out


# ===================================================================
# Noise channels
# ===================================================================
class BitFlip(KrausChannel):
    r"""
    Single-qubit bit-flip (Pauli X) error channel.

        K₀ = √(1-p) I,   K₁ = √p X

    where p ∈ [0, 1] is the probability of a bit flip.
    """

    def __init__(self, p: float, wires: Union[int, List[int]] = 0):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        p = self.p
        K0 = jnp.sqrt(1 - p) * I.astype(jnp.complex128)
        K1 = jnp.sqrt(p) * X.astype(jnp.complex128)
        return [K0, K1]


class PhaseFlip(KrausChannel):
    r"""
    Single-qubit phase-flip (Pauli Z) error channel.

        K₀ = √(1-p) I,   K₁ = √p Z

    where p ∈ [0, 1] is the probability of a phase flip.
    """

    def __init__(self, p: float, wires: Union[int, List[int]] = 0):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        p = self.p
        K0 = jnp.sqrt(1 - p) * I.astype(jnp.complex128)
        K1 = jnp.sqrt(p) * Z.astype(jnp.complex128)
        return [K0, K1]


class DepolarizingChannel(KrausChannel):
    r"""
    Single-qubit depolarizing channel.

        K₀ = √(1-p) I,   K₁ = √(p/3) X,
        K₂ = √(p/3) Y,   K₃ = √(p/3) Z

    where p ∈ [0, 1]. At p=3/4 the channel is fully depolarizing.
    """

    def __init__(self, p: float, wires: Union[int, List[int]] = 0):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        p = self.p
        K0 = jnp.sqrt(1 - p) * I.astype(jnp.complex128)
        K1 = jnp.sqrt(p / 3) * X.astype(jnp.complex128)
        K2 = jnp.sqrt(p / 3) * Y
        K3 = jnp.sqrt(p / 3) * Z.astype(jnp.complex128)
        return [K0, K1, K2, K3]


class AmplitudeDamping(KrausChannel):
    r"""
    Single-qubit amplitude damping channel.

        K₀ = [[1,        0      ],      K₁ = [[0, √γ],
              [0, √(1-γ)       ]]             [0,  0 ]]

    where γ ∈ [0, 1] is the probability of energy loss (|1⟩→|0⟩).
    """

    def __init__(self, gamma: float, wires: Union[int, List[int]] = 0):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.gamma = gamma
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        g = self.gamma
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=jnp.complex128)
        K1 = jnp.array([[0.0, jnp.sqrt(g)], [0.0, 0.0]], dtype=jnp.complex128)
        return [K0, K1]


class PhaseDamping(KrausChannel):
    r"""
    Single-qubit phase damping (dephasing) channel.

        K₀ = [[1,        0      ],      K₁ = [[0,      0     ],
              [0, √(1-γ)       ]]             [0, √γ         ]]

    where γ ∈ [0, 1] is the phase damping probability.
    """

    def __init__(self, gamma: float, wires: Union[int, List[int]] = 0):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.gamma = gamma
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        g = self.gamma
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=jnp.complex128)
        K1 = jnp.array([[0.0, 0.0], [0.0, jnp.sqrt(g)]], dtype=jnp.complex128)
        return [K0, K1]


class ThermalRelaxationError(KrausChannel):
    r"""
    Single-qubit thermal relaxation error channel.

    Models simultaneous T₁ energy relaxation and T₂ dephasing.
    Two regimes:

    **T₂ ≤ T₁** (Markovian dephasing + reset):

        Six Kraus operators built from p_z (phase-flip prob), p_r0 (reset-to-0
        prob) and p_r1 (reset-to-1 prob).

    **T₂ > T₁** (non-Markovian; Choi matrix decomposition):

        Choi matrix is built from the relaxation/dephasing rates, then
        diagonalised; Kraus operators are K_i = √λ_i · mat(v_i).

    Parameters
    ----------
    pe : float
        Excited-state population (thermal population of |1⟩), ∈ [0, 1].
    t1 : float
        T₁ longitudinal relaxation time (> 0).
    t2 : float
        T₂ transverse dephasing time (> 0, ≤ 2 T₁).
    tg : float
        Gate duration (> 0).
    """

    def __init__(
        self,
        pe: float,
        t1: float,
        t2: float,
        tg: float,
        wires: Union[int, List[int]] = 0,
    ):
        if not 0.0 <= pe <= 1.0:
            raise ValueError("pe must be in [0, 1].")
        if t1 <= 0:
            raise ValueError("t1 must be > 0.")
        if t2 <= 0:
            raise ValueError("t2 must be > 0.")
        if t2 > 2 * t1:
            raise ValueError("t2 must be ≤ 2·t1.")
        if tg < 0:
            raise ValueError("tg must be ≥ 0.")
        self.pe = pe
        self.t1 = t1
        self.t2 = t2
        self.tg = tg
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        pe, t1, t2, tg = self.pe, self.t1, self.t2, self.tg

        eT1 = jnp.exp(-tg / t1)
        p_reset = 1.0 - eT1
        eT2 = jnp.exp(-tg / t2)

        if t2 <= t1:
            # --- Case T₂ ≤ T₁: six Kraus operators ---
            pz = (1.0 - p_reset) * (1.0 - eT2 / eT1) / 2.0
            pr0 = (1.0 - pe) * p_reset
            pr1 = pe * p_reset
            pid = 1.0 - pz - pr0 - pr1

            K0 = jnp.sqrt(pid) * jnp.eye(2, dtype=jnp.complex128)
            K1 = jnp.sqrt(pz) * jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
            K2 = jnp.sqrt(pr0) * jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
            K3 = jnp.sqrt(pr0) * jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)
            K4 = jnp.sqrt(pr1) * jnp.array([[0, 0], [1, 0]], dtype=jnp.complex128)
            K5 = jnp.sqrt(pr1) * jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)
            return [K0, K1, K2, K3, K4, K5]

        else:
            # --- Case T₂ > T₁: Choi matrix decomposition ---
            # Choi matrix (column-major / reshaping convention matching PennyLane)
            choi = jnp.array(
                [
                    [1 - pe * p_reset, 0, 0, eT2],
                    [0, pe * p_reset, 0, 0],
                    [0, 0, (1 - pe) * p_reset, 0],
                    [eT2, 0, 0, 1 - (1 - pe) * p_reset],
                ],
                dtype=jnp.complex128,
            )
            eigenvalues, eigenvectors = jnp.linalg.eigh(choi)
            # Each eigenvector (column of eigenvectors) reshaped as 2×2 → one Kraus op
            kraus = []
            for i in range(4):
                lam = eigenvalues[i]
                vec = eigenvectors[:, i]
                # Map ℂ⁴ → ℂ²ˣ² with column-major order (matching PL convention)
                mat = jnp.sqrt(jnp.abs(lam)) * vec.reshape(2, 2, order="F")  # type: ignore[call-overload]
                kraus.append(mat.astype(jnp.complex128))
            return kraus


class RX(Operation):
    """Rotation around X: RX(θ) = exp(-i θ/2 X)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0):
        self.theta = theta
        mat = jnp.cos(theta / 2) * I - 1j * jnp.sin(theta / 2) * X
        super().__init__(wires=wires, matrix=mat)


class RY(Operation):
    """Rotation around Y: RY(θ) = exp(-i θ/2 Y)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0):
        self.theta = theta
        mat = jnp.cos(theta / 2) * I - 1j * jnp.sin(theta / 2) * Y
        super().__init__(wires=wires, matrix=mat)


class RZ(Operation):
    """Rotation around Z: RZ(θ) = exp(-i θ/2 Z)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0):
        self.theta = theta
        mat = jnp.cos(theta / 2) * I - 1j * jnp.sin(theta / 2) * Z
        super().__init__(wires=wires, matrix=mat)
