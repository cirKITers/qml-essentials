from __future__ import annotations

import threading
from typing import Callable, List, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg

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
        Apply this gate directly to a statevector without ever building the
        full 2**n × 2**n unitary matrix.

        The statevector (shape ``(2**n,)``) is reshaped into a rank-*n* tensor
        of shape ``(2, 2, …, 2)``.  The gate matrix (shape ``(2**k, 2**k)`` for
        a *k*-qubit gate) is reshaped to ``(2,)*2k`` and contracted against the
        *k* target axes via ``jnp.tensordot``.  The resulting tensor is then
        moved back to axis order ``(0, 1, …, n-1)`` and flattened.

        Memory:  O(2**n)  — no quadratic allocation.
        Supports arbitrary k-qubit gates on arbitrary wire subsets.
        Fully differentiable through JAX.
        """
        k = len(self.wires)
        gate_tensor = self.matrix.reshape((2,) * 2 * k)

        # Reshape state: (2**n,) → (2,)*n
        psi = state.reshape((2,) * n_qubits)

        # Contract gate_tensor over its *in* axes (last k axes) with the target
        # wire axes of psi.  tensordot sums over axes: gate[..., i0,i1,...] * psi[...,i0,i1,...]
        # Result axes: (gate out-axes) + (remaining psi axes in original order)
        psi_out = jnp.tensordot(
            gate_tensor, psi, axes=(list(range(k, 2 * k)), self.wires)
        )

        # tensordot puts the k new (output) axes first, then the n-k untouched axes.
        # We need to move the k output axes back to their original wire positions.
        # Current axis layout after tensordot:
        #   [new_wire_0, new_wire_1, ..., new_wire_{k-1},  <remaining axes in original order>]
        # We want:  [qubit_0, qubit_1, ..., qubit_{n-1}]
        remaining = [q for q in range(n_qubits) if q not in self.wires]
        # Source positions of each target qubit in the tensordot output
        source = list(range(k)) + list(range(k, n_qubits))
        # Destination positions we want them at
        dest = list(self.wires) + remaining
        # Build inverse permutation: dest[i] should go to position dest[i]
        perm = [0] * n_qubits
        for src_pos, dst_pos in zip(source, dest):
            perm[dst_pos] = src_pos
        psi_out = jnp.transpose(psi_out, perm)

        return psi_out.reshape(2**n_qubits)

    def full_matrix(self, n_qubits: int) -> jnp.ndarray:
        """
        Expand the gate into the full ``2**n × 2**n`` unitary by applying it
        to each computational basis vector.

        This is kept as a convenience method for observable lifting and
        small-system inspection, but ``apply_to_state`` should be preferred
        for circuit execution as it avoids the quadratic memory allocation.
        """
        dim = 2**n_qubits
        basis = jnp.eye(dim, dtype=jnp.complex128)
        columns = jax.vmap(lambda col: self.apply_to_state(col, n_qubits))(basis)
        # vmap maps over rows of `basis`, so columns[i] = U|i⟩ → U = columns.T
        return columns.T


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
# RX / RY / RZ  – parameterized rotation gates
# ===================================================================
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


# ===================================================================
# evolve – Hamiltonian time evolution
# ===================================================================
def evolve(hermitian: Hermitian):
    """
    Returns a callable gate-factory that applies the unitary

        U(t) = exp(-i t H)

    for a given Hermitian operator *H*.  The returned callable accepts
    a scalar parameter ``t`` and optional ``wires``.

    This is fully differentiable through ``jax.grad`` because
    ``jax.scipy.linalg.expm`` supports reverse-mode AD.

    Example::

        time_evol = evolve(Hermitian(matrix=sigma_z, wires=0))
        time_evol(t=0.5, wires=0)           # inside a circuit
    """
    H_mat = hermitian.matrix

    def _apply(t: float, wires: Union[int, List[int]] = 0):
        U = jax.scipy.linalg.expm(-1j * t * H_mat)
        return type("EvolvedOp", (Operation,), {})(wires=wires, matrix=U)

    return _apply


# ===================================================================
# QuantumScript – circuit container & executor
# ===================================================================
class QuantumScript:
    """
    This forms the basis for any quantum circuit.

    It takes a callable *f* which is the circuit to execute.
    Within *f*, different operations are instantiated and automatically
    recorded onto a tape.

    Parameters
    ----------
    f : callable
        A function whose body instantiates Operation objects.
        Signature:  ``f(*args, **kwargs) -> None``
    n_qubits : int or None
        Number of qubits. If *None*, inferred from the operations.
    """

    def __init__(self, f: Callable, n_qubits: Optional[int] = None):
        self.f = f
        self._n_qubits = n_qubits

    # -- internal: record the tape by running f -------------------------
    def _record(self, *args, **kwargs) -> List[Operation]:
        tape: List[Operation] = []
        _set_tape(tape)
        try:
            self.f(*args, **kwargs)
        finally:
            _set_tape(None)
        return tape

    # -- infer qubit count from the tape --------------------------------
    @staticmethod
    def _infer_n_qubits(ops: List[Operation], obs: List[Operation]) -> int:
        all_wires: set[int] = set()
        for op in ops + obs:
            all_wires.update(op.wires)
        return max(all_wires) + 1 if all_wires else 1

    # -- core execution -------------------------------------------------
    def execute(
        self,
        type: str = "expval",
        obs: Optional[List[Operation]] = None,
        *,
        args: tuple = (),
        kwargs: dict = {},
    ) -> jnp.ndarray:
        """
        Execute the circuit and return measurement results.

        Parameters
        ----------
        type : str
            ``"expval"`` - expectation value ⟨ψ|O|ψ⟩ for each observable.
            ``"probs"``  - probability vector |⟨i|ψ⟩|² .
            ``"state"``  - raw statevector.
        obs : list[Operation]
            Observables (only used for ``"expval"``).

        Returns
        -------
        jnp.ndarray
        """
        if obs is None:
            obs = []

        # 1. Record the operations
        tape = self._record(*args, **kwargs)

        # 2. Determine system size
        n_qubits = self._n_qubits or self._infer_n_qubits(tape, obs)
        dim = 2**n_qubits

        # 3. Build initial state |00…0⟩
        state = jnp.zeros(dim, dtype=jnp.complex128)
        state = state.at[0].set(1.0)

        # 4. Apply each gate via tensor contraction (no full-matrix allocation)
        for op in tape:
            state = op.apply_to_state(state, n_qubits)

        # 5. Measurement
        if type == "state":
            return state

        if type == "probs":
            return jnp.abs(state) ** 2

        if type == "expval":
            results = []
            for ob in obs:
                O = ob.full_matrix(n_qubits)
                ev = jnp.real(jnp.conj(state) @ O @ state)
                results.append(ev)
            return jnp.array(results)

        raise ValueError(f"Unknown measurement type: {type!r}")
