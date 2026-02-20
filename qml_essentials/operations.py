from typing import Callable, List, Optional, Tuple, Union
from functools import lru_cache

import jax
import jax.numpy as jnp

from qml_essentials.tape import active_tape, recording  # noqa: F401 (re-export)


@lru_cache(maxsize=256)
def _permutation_for_contraction(
    total: int,
    k: int,
    target_axes: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Pre-compute the contraction and permutation indices.

    These are pure functions of ``(total, k, target_axes)`` and never change
    for a given gate/wire combination.  Caching them avoids rebuilding
    Python lists on every single gate application — a measurable overhead
    when circuits have hundreds of gates.

    Args:
        total: Total rank of the tensor (``n`` for states, ``2n`` for density
            matrices).
        k: Number of qubits the gate acts on.
        target_axes: The axes to contract against (as a tuple for hashability).

    Returns:
        ``(contract_axes, perm)`` — the axes to pass to ``jnp.tensordot`` and
        the permutation to restore the original axis order.
    """
    contract_axes = tuple(range(k, 2 * k))
    target_set = set(target_axes)
    remaining = [ax for ax in range(total) if ax not in target_set]
    dest = list(target_axes) + remaining
    perm = [0] * total
    for src_pos, dst_pos in enumerate(dest):
        perm[dst_pos] = src_pos
    return contract_axes, tuple(perm)


def _contract_and_restore(
    tensor: jnp.ndarray,
    gate: jnp.ndarray,
    k: int,
    target_axes: List[int],
) -> jnp.ndarray:
    """Contract *gate* against *target_axes* of *tensor* and restore axis order.

    This is the core building block for applying a ``k``-qubit gate tensor to
    a rank-*n* (statevector) or rank-*2n* (density-matrix) tensor.  After
    ``jnp.tensordot``, the output axes are permuted back so that the
    contracted axes appear in their original positions.

    The permutation indices are cached via :func:`_permutation_for_contraction`
    so that the Python-level list construction only happens once per unique
    ``(total, k, target_axes)`` combination.

    Args:
        tensor: Rank-*N* tensor (e.g. ``(2,)*n`` for states or ``(2,)*2n``
            for density matrices).
        gate: Reshaped gate tensor of shape ``(2,)*2k``.
        k: Number of qubits the gate acts on (= ``len(target_axes)``).
        target_axes: The *k* axes of *tensor* to contract against.

    Returns:
        Updated tensor with the same rank as *tensor*, with the
        contracted axes restored to their original positions.
    """
    contract_axes, perm = _permutation_for_contraction(
        tensor.ndim, k, tuple(target_axes)
    )
    out = jnp.tensordot(gate, tensor, axes=(contract_axes, target_axes))
    return jnp.transpose(out, perm)


# ===================================================================
# Operation base class
# ===================================================================
class Operation:
    """Base class for any quantum operation or observable.

    Further gates should inherit from this class to realise more specific
    operations.  Generally, operations are created by instantiation inside a
    circuit function passed to :class:`QuantumScript`; the instance is
    automatically appended to the active tape.

    An ``Operation`` can also serve as an *observable*: its matrix is used to
    compute expectation values via ``apply_to_state`` / ``apply_to_density``.

    Attributes:
        _matrix: Class-level default gate matrix.  Subclasses set this to their
            fixed unitary.  Instances may override it via the *matrix* argument
            to ``__init__``.
    """

    # Subclasses should set this to the gate's unitary / matrix
    _matrix: jnp.ndarray = None
    is_controlled = False

    def __init__(
        self,
        wires: Union[int, List[int]] = 0,
        matrix: Optional[jnp.ndarray] = None,
        record: bool = True,
    ) -> None:
        """Initialise the operation and optionally register it on the active tape.

        Args:
            wires: Qubit index or list of qubit indices this operation acts on.
            matrix: Optional explicit gate matrix.  When provided it overrides
                the class-level ``_matrix`` attribute.
            record: If ``True`` (default) and a tape is currently recording,
                append this operation to the tape.  Set to ``False`` for
                auxiliary objects that should not appear in the circuit
                (e.g. Hamiltonians used only to build time-dependent
                evolutions).
        """
        self.wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]
        if matrix is not None:
            self._matrix = matrix

        # If a tape is currently recording, append ourselves
        if record:
            tape = active_tape()
            if tape is not None:
                tape.append(self)

    @property
    def name(self) -> str:
        """Return the class name of this operation (e.g. ``'RX'``, ``'CX'``).

        Returns:
            The operation name string.
        """
        return self.__class__.__name__

    @property
    def parameters(self) -> list:
        """Return the list of numeric parameters for this operation.

        Parametrized gates (RX, RY, RZ, CRX, CRY, CRZ, Rot) store their
        angles as instance attributes.  This property collects them in a
        canonical order.  Non-parametrized gates return an empty list.

        Returns:
            List of parameter values (floats or JAX arrays).
        """
        # Parametrized single-qubit rotations
        if hasattr(self, "theta"):
            return [self.theta]
        # Rot gate has three parameters
        if hasattr(self, "phi") and hasattr(self, "omega"):
            return [self.phi, self.theta, self.omega]
        # KrausChannel with p
        if hasattr(self, "p"):
            return [self.p]
        return []

    @property
    def matrix(self) -> jnp.ndarray:
        """Return the base matrix of this operation (before lifting).

        Returns:
            The gate matrix as a JAX array.

        Raises:
            NotImplementedError: If the subclass has not defined ``_matrix``.
        """
        if self._matrix is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not define a matrix."
            )
        return self._matrix

    @property
    def wires(self) -> List[int]:
        """Qubit indices this operation acts on.

        Returns:
            List of integer qubit indices.
        """
        return self._wires

    @wires.setter
    def wires(self, wires: Union[int, List[int]]) -> None:
        """Set the qubit indices for this operation.

        Args:
            wires: A single qubit index or a list of qubit indices.
        """
        if isinstance(wires, (list, tuple)):
            self._wires = list(wires)
        else:
            self._wires = [wires]

    def lifted_matrix(self, n_qubits: int) -> jnp.ndarray:
        """Return the full ``2**n x 2**n`` matrix embedding this gate.

        Embeds the ``k``-qubit gate matrix into the ``n``-qubit Hilbert space
        by applying it to the identity matrix via :meth:`apply_to_state`.
        This is useful for computing ``Tr(O·ρ)`` directly without vmap.

        Args:
            n_qubits: Total number of qubits in the circuit.

        Returns:
            The ``(2**n, 2**n)`` matrix of this operation in the full space.
        """
        dim = 2**n_qubits
        # Apply the gate to each basis vector (column of identity)
        return jax.vmap(lambda col: self.apply_to_state(col, n_qubits))(
            jnp.eye(dim, dtype=jnp.complex128)
        ).T

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a statevector via tensor contraction.

        The statevector (shape ``(2**n,)``) is reshaped into a rank-*n* tensor
        of shape ``(2,)*n``.  The gate (shape ``(2**k, 2**k)``) is reshaped to
        ``(2,)*2k`` and contracted against the *k* target wire axes.

        Memory footprint is O(2**n) and the operation supports arbitrary *k*.
        The implementation is fully differentiable through JAX.

        Args:
            state: Statevector of shape ``(2**n_qubits,)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated statevector of shape ``(2**n_qubits,)``.
        """
        k = len(self.wires)
        gate_tensor = self.matrix.reshape((2,) * 2 * k)
        psi = state.reshape((2,) * n_qubits)
        psi_out = _contract_and_restore(psi, gate_tensor, k, self.wires)
        return psi_out.reshape(2**n_qubits)

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a density matrix via ρ → UρU†.

        The density matrix (shape ``(2**n, 2**n)``) is treated as a rank-*2n*
        tensor with *n* "ket" axes (0..n-1) and *n* "bra" axes (n..2n-1).
        U acts on the ket half; U* acts on the bra half.  Both contractions
        use the shared :func:`_contract_and_restore` helper, keeping the
        operation allocation-free with respect to building full unitaries.

        Args:
            rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        k = len(self.wires)
        U = self.matrix.reshape((2,) * 2 * k)
        U_conj = jnp.conj(U)

        rho_t = rho.reshape((2,) * 2 * n_qubits)

        # Apply U to ket axes, U† to bra axes
        rho_t = _contract_and_restore(rho_t, U, k, self.wires)
        bra_wires = [w + n_qubits for w in self.wires]
        rho_t = _contract_and_restore(rho_t, U_conj, k, bra_wires)

        return rho_t.reshape(2**n_qubits, 2**n_qubits)


# ===================================================================
# Concrete gates / observables
# ===================================================================
class Hermitian(Operation):
    """A generic Hermitian observable or gate defined by an arbitrary matrix.

    Example:
        >>> obs = Hermitian(matrix=my_matrix, wires=0)
    """

    def __init__(
        self,
        matrix: jnp.ndarray,
        wires: Union[int, List[int]] = 0,
        record: bool = True,
    ) -> None:
        """Initialise a Hermitian operator.

        Args:
            matrix: The Hermitian matrix defining this operator.
            wires: Qubit index or list of qubit indices this operator acts on.
            record: If ``True`` (default), record on the active tape.  Set to
                ``False`` when using the Hermitian purely as a Hamiltonian
                component (e.g. for time-dependent evolution).
        """
        super().__init__(
            wires=wires,
            matrix=jnp.asarray(matrix, dtype=jnp.complex128),
            record=record,
        )

    def __rmul__(self, coeff_fn):
        """Support ``coeff_fn * Hermitian`` → :class:`ParametrizedHamiltonian`.

        Args:
            coeff_fn: A callable ``(params, t) -> scalar`` giving the
                time-dependent coefficient.

        Returns:
            A :class:`ParametrizedHamiltonian` pairing *coeff_fn* with this
            operator's matrix and wires.

        Raises:
            TypeError: If *coeff_fn* is not callable.
        """
        if not callable(coeff_fn):
            raise TypeError(
                f"Left operand of `* Hermitian` must be callable, got {type(coeff_fn)}"
            )
        return ParametrizedHamiltonian(coeff_fn, self.matrix, self.wires)


class ParametrizedHamiltonian:
    """A time-dependent Hamiltonian ``H(t) = f(params, t) · H_mat``.

    Created by multiplying a callable coefficient function with a
    :class:`Hermitian` operator::

        def coeff(p, t):
            return p[0] * jnp.exp(-0.5 * ((t - t_c) / p[1]) ** 2)

        H_td = coeff * Hermitian(matrix=sigma_x, wires=0)

    The Hamiltonian is then used with :func:`evolve`::

        evolve(H_td)(coeff_args=[A, sigma], T=1.0)

    Attributes:
        coeff_fn: Callable ``(params, t) -> scalar``.
        H_mat: Static Hermitian matrix (JAX array).
        wires: Qubit wire(s) this Hamiltonian acts on.
    """

    def __init__(
        self,
        coeff_fn: Callable,
        H_mat: jnp.ndarray,
        wires: Union[int, List[int]],
    ) -> None:
        self.coeff_fn = coeff_fn
        self.H_mat = H_mat
        self.wires = wires


class Id(Operation):
    """Identity gate."""

    _matrix = jnp.eye(2, dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise an identity gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class PauliX(Operation):
    """Pauli-X gate / observable (bit-flip, σ_x)."""

    _matrix = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-X gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class PauliY(Operation):
    """Pauli-Y gate / observable (σ_y)."""

    _matrix = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-Y gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class PauliZ(Operation):
    """Pauli-Z gate / observable (phase-flip, σ_z)."""

    _matrix = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-Z gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class H(Operation):
    """Hadamard gate."""

    _matrix = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Hadamard gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class S(Operation):
    """S (phase) gate — a Clifford gate equal to √Z.

    .. math::
        S = \\begin{pmatrix}1 & 0\\\\ 0 & i\\end{pmatrix}
    """

    _matrix = jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an S gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class Barrier(Operation):
    """Barrier operation — a no-op used for visual circuit separation.

    The barrier does not change the quantum state.  It is recorded on the
    tape so that drawing backends can insert a visual separator.
    """

    _matrix = None  # not a real gate

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Barrier.

        Args:
            wires: Qubit index or list of qubit indices this barrier spans.
        """
        super().__init__(wires=wires)

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """No-op: return the state unchanged."""
        return state

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """No-op: return the density matrix unchanged."""
        return rho


class RX(Operation):
    """Rotation around the X axis: RX(θ) = exp(-i θ/2 X)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an RX rotation gate.

        Args:
            theta: Rotation angle in radians.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.theta = theta
        mat = jnp.cos(theta / 2) * Id._matrix - 1j * jnp.sin(theta / 2) * PauliX._matrix
        super().__init__(wires=wires, matrix=mat)

    def generator(self) -> Operation:
        """Return the generator ``-0.5 · X`` as a :class:`PauliX` operation."""
        return PauliX(wires=self.wires[0], record=False)


class RY(Operation):
    """Rotation around the Y axis: RY(θ) = exp(-i θ/2 Y)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an RY rotation gate.

        Args:
            theta: Rotation angle in radians.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.theta = theta
        mat = jnp.cos(theta / 2) * Id._matrix - 1j * jnp.sin(theta / 2) * PauliY._matrix
        super().__init__(wires=wires, matrix=mat)

    def generator(self) -> Operation:
        """Return the generator ``-0.5 · Y`` as a :class:`PauliY` operation."""
        return PauliY(wires=self.wires[0], record=False)


class RZ(Operation):
    """Rotation around the Z axis: RZ(θ) = exp(-i θ/2 Z)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an RZ rotation gate.

        Args:
            theta: Rotation angle in radians.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.theta = theta
        mat = jnp.cos(theta / 2) * Id._matrix - 1j * jnp.sin(theta / 2) * PauliZ._matrix
        super().__init__(wires=wires, matrix=mat)

    def generator(self) -> Operation:
        """Return the generator ``-0.5 · Z`` as a :class:`PauliZ` operation."""
        return PauliZ(wires=self.wires[0], record=False)


class CX(Operation):
    """Controlled-X (CNOT) gate.

    Args on construction:
        wires: ``[control, target]``.
    """

    _matrix = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=jnp.complex128,
    )
    is_controlled = True

    def __init__(self, wires: List[int] = [0, 1]) -> None:
        """Initialise a Controlled-X gate.

        Args:
            wires: Two-element list ``[control, target]``.
        """
        super().__init__(wires=wires)


class CCX(Operation):
    """Toffoli (CCX) gate.

    The 3-qubit Toffoli gate exercises the arbitrary-k-qubit path in
    :meth:`~Operation.apply_to_state` and cannot be expressed as a pair of
    2-qubit gates without ancilla, making it a good stress-test for the
    simulator.
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
    is_controlled = True

    def __init__(self, wires: List[int] = [0, 1, 2]) -> None:
        """Initialise a Toffoli (CCX) gate.

        Args:
            wires: Three-element list ``[control0, control1, target]``.
        """
        super().__init__(wires=wires)


class CSWAP(Operation):
    """Controlled-SWAP (Fredkin) gate.

    Swaps the two target qubits conditioned on the control qubit being |1⟩.

    Args on construction:
        wires: ``[control, target0, target1]``.
    """

    _matrix = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=jnp.complex128,
    )
    is_controlled = True

    def __init__(self, wires: List[int] = [0, 1, 2]) -> None:
        """Initialise a Controlled-SWAP (Fredkin) gate.

        Args:
            wires: Three-element list ``[control, target0, target1]``.
        """
        super().__init__(wires=wires)


class CY(Operation):
    """Controlled-Y gate.

    Applies a Pauli-Y gate on the target qubit conditioned on the control
    qubit being in state |1⟩.

    Args on construction:
        wires: ``[control, target]``.
    """

    _matrix = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dtype=jnp.complex128,
    )
    is_controlled = True

    def __init__(self, wires: List[int] = [0, 1]) -> None:
        """Initialise a Controlled-Y gate.

        Args:
            wires: Two-element list ``[control, target]``.
        """
        super().__init__(wires=wires)


class CZ(Operation):
    """Controlled-Z gate.

    Applies a Pauli-Z gate on the target qubit conditioned on the control
    qubit being in state |1⟩.

    Args on construction:
        wires: ``[control, target]``.
    """

    _matrix = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dtype=jnp.complex128,
    )
    is_controlled = True

    def __init__(self, wires: List[int] = [0, 1]) -> None:
        """Initialise a Controlled-Z gate.

        Args:
            wires: Two-element list ``[control, target]``.
        """
        super().__init__(wires=wires)


class CRX(Operation):
    """Controlled rotation around the X axis.

    Applies RX(θ) on the target qubit conditioned on the control qubit
    being in state |1⟩.

    .. math::
        CRX(\\theta) = |0\\rangle\\langle 0| \\otimes I
                      + |1\\rangle\\langle 1| \\otimes RX(\\theta)
    """

    is_controlled = True

    def __init__(self, theta: float, wires: List[int] = [0, 1]) -> None:
        """Initialise a CRX gate.

        Args:
            theta: Rotation angle in radians.
            wires: Two-element list ``[control, target]``.
        """
        self.theta = theta
        c = jnp.cos(theta / 2)
        s = jnp.sin(theta / 2)
        mat = jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, c, -1j * s],
                [0, 0, -1j * s, c],
            ],
            dtype=jnp.complex128,
        )
        super().__init__(wires=wires, matrix=mat)


class CRY(Operation):
    """Controlled rotation around the Y axis.

    Applies RY(θ) on the target qubit conditioned on the control qubit
    being in state |1⟩.

    .. math::
        CRY(\\theta) = |0\\rangle\\langle 0| \\otimes I
                      + |1\\rangle\\langle 1| \\otimes RY(\\theta)
    """

    is_controlled = True

    def __init__(self, theta: float, wires: List[int] = [0, 1]) -> None:
        """Initialise a CRY gate.

        Args:
            theta: Rotation angle in radians.
            wires: Two-element list ``[control, target]``.
        """
        self.theta = theta
        c = jnp.cos(theta / 2)
        s = jnp.sin(theta / 2)
        mat = jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, c, -s],
                [0, 0, s, c],
            ],
            dtype=jnp.complex128,
        )
        super().__init__(wires=wires, matrix=mat)


class CRZ(Operation):
    """Controlled rotation around the Z axis.

    Applies RZ(θ) on the target qubit conditioned on the control qubit
    being in state |1⟩.

    .. math::
        CRZ(\\theta) = |0\\rangle\\langle 0| \\otimes I
                      + |1\\rangle\\langle 1| \\otimes RZ(\\theta)
    """

    is_controlled = True

    def __init__(self, theta: float, wires: List[int] = [0, 1]) -> None:
        """Initialise a CRZ gate.

        Args:
            theta: Rotation angle in radians.
            wires: Two-element list ``[control, target]``.
        """
        self.theta = theta
        mat = jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, jnp.exp(-1j * theta / 2), 0],
                [0, 0, 0, jnp.exp(1j * theta / 2)],
            ],
            dtype=jnp.complex128,
        )
        super().__init__(wires=wires, matrix=mat)


class Rot(Operation):
    """General single-qubit rotation: Rot(φ, θ, ω) = RZ(ω) RY(θ) RZ(φ).

    This is the most general SU(2) rotation (up to a global phase).  It
    decomposes into three successive rotations and has three free parameters.
    """

    def __init__(
        self,
        phi: float,
        theta: float,
        omega: float,
        wires: Union[int, List[int]] = 0,
    ) -> None:
        """Initialise a general rotation gate.

        Args:
            phi: First RZ rotation angle (radians).
            theta: RY rotation angle (radians).
            omega: Second RZ rotation angle (radians).
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.phi = phi
        self.theta = theta
        self.omega = omega
        # Rot(φ, θ, ω) = RZ(ω) @ RY(θ) @ RZ(φ)
        rz_phi = jnp.cos(phi / 2) * Id._matrix - 1j * jnp.sin(phi / 2) * PauliZ._matrix
        ry_theta = (
            jnp.cos(theta / 2) * Id._matrix - 1j * jnp.sin(theta / 2) * PauliY._matrix
        )
        rz_omega = (
            jnp.cos(omega / 2) * Id._matrix - 1j * jnp.sin(omega / 2) * PauliZ._matrix
        )
        mat = rz_omega @ ry_theta @ rz_phi
        super().__init__(wires=wires, matrix=mat)


class PauliRot(Operation):
    """Multi-qubit Pauli rotation: exp(-i θ/2 P) for a Pauli word P.

    The Pauli word is given as a string of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
    characters (one per qubit).  The rotation matrix is computed as
    ``cos(θ/2) I - i sin(θ/2) P`` where *P* is the tensor product of the
    corresponding single-qubit Pauli matrices.

    Example::

        PauliRot(0.5, "XY", wires=[0, 1])
    """

    # Map from character to 2x2 matrix
    _PAULI_MAP = {
        "I": Id._matrix,
        "X": PauliX._matrix,
        "Y": PauliY._matrix,
        "Z": PauliZ._matrix,
    }

    def __init__(
        self,
        theta: float,
        pauli_word: str,
        wires: Union[int, List[int]] = 0,
    ) -> None:
        """Initialise a PauliRot gate.

        Args:
            theta: Rotation angle in radians.
            pauli_word: A string of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
                characters specifying the Pauli tensor product.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        from functools import reduce as _reduce

        self.theta = theta
        self.pauli_word = pauli_word

        pauli_matrices = [self._PAULI_MAP[c] for c in pauli_word]
        P = _reduce(jnp.kron, pauli_matrices)
        dim = P.shape[0]
        mat = (
            jnp.cos(theta / 2) * jnp.eye(dim, dtype=jnp.complex128)
            - 1j * jnp.sin(theta / 2) * P
        )
        super().__init__(wires=wires, matrix=mat)

    def generator(self) -> Operation:
        """Return the generator Pauli tensor product as an :class:`Operation`.

        The generator of ``PauliRot(θ, word, wires)`` is the tensor product
        of single-qubit Pauli matrices specified by *word*.  The returned
        :class:`Hermitian` wraps that matrix and the gate's wires.

        Returns:
            :class:`Hermitian` operation representing the Pauli tensor product.
        """
        from functools import reduce as _reduce

        pauli_matrices = [self._PAULI_MAP[c] for c in self.pauli_word]
        P = _reduce(jnp.kron, pauli_matrices)
        return Hermitian(matrix=P, wires=self.wires, record=False)


# ===================================================================
# Kraus channel base class
# ===================================================================
class KrausChannel(Operation):
    """Base class for noise channels defined by a set of Kraus operators.

    A Kraus channel Φ(ρ) = Σ_k K_k ρ K_k† is the most general physical
    operation on a quantum state.  For a pure unitary gate there is a single
    operator K_0 = U satisfying K_0†K_0 = I; for noisy channels there are
    multiple operators.

    Subclasses must implement :meth:`kraus_matrices` and return a list of JAX
    arrays.  :meth:`apply_to_state` is intentionally left unimplemented:
    Kraus channels require a density-matrix representation and cannot be
    applied to a pure statevector in general.
    """

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the list of Kraus operators for this channel.

        Returns:
            List of 2-D JAX arrays, each of shape ``(2**k, 2**k)`` where *k*
            is the number of target qubits.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    @property
    def matrix(self) -> jnp.ndarray:
        """Raises TypeError — noise channels have no single unitary matrix.

        Raises:
            TypeError: Always raised; use :meth:`apply_to_density` instead.
        """
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and has no single "
            "unitary matrix. Use apply_to_density() instead."
        )

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Raises TypeError — noise channels require density-matrix simulation.

        Args:
            state: Statevector (unused).
            n_qubits: Number of qubits (unused).

        Raises:
            TypeError: Always raised; use ``execute(type='density')`` instead.
        """
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and cannot be "
            "applied to a pure statevector. Use execute(type='density') instead."
        )

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply Φ(ρ) = Σ_k K_k ρ K_k† using tensor-contraction.

        Uses the shared :func:`_contract_and_restore` helper, summing the
        result over all Kraus operators.

        Args:
            rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        k = len(self.wires)
        dim = 2**n_qubits
        bra_wires = [w + n_qubits for w in self.wires]
        rho_out = jnp.zeros_like(rho)

        for K in self.kraus_matrices():
            K_t = K.reshape((2,) * 2 * k)
            K_conj_t = jnp.conj(K_t)
            rho_t = rho.reshape((2,) * 2 * n_qubits)
            rho_t = _contract_and_restore(rho_t, K_t, k, self.wires)
            rho_t = _contract_and_restore(rho_t, K_conj_t, k, bra_wires)
            rho_out = rho_out + rho_t.reshape(dim, dim)

        return rho_out


# ===================================================================
# Noise channels
# ===================================================================
class BitFlip(KrausChannel):
    r"""Single-qubit bit-flip (Pauli-X) error channel.

    .. math::
        K_0 = \sqrt{1-p}\,I, \quad K_1 = \sqrt{p}\,X

    where *p* ∈ [0, 1] is the probability of a bit flip.
    """

    def __init__(self, p: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a bit-flip channel.

        Args:
            p: Bit-flip probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *p* is outside [0, 1].
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the bit-flip channel.

        Returns:
            List ``[K0, K1]`` where K0 = √(1-p)·I and K1 = √p·X.
        """
        p = self.p
        K0 = jnp.sqrt(1 - p) * Id._matrix
        K1 = jnp.sqrt(p) * PauliX._matrix
        return [K0, K1]


class PhaseFlip(KrausChannel):
    r"""Single-qubit phase-flip (Pauli-Z) error channel.

    .. math::
        K_0 = \sqrt{1-p}\,I, \quad K_1 = \sqrt{p}\,Z

    where *p* ∈ [0, 1] is the probability of a phase flip.
    """

    def __init__(self, p: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a phase-flip channel.

        Args:
            p: Phase-flip probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *p* is outside [0, 1].
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the phase-flip channel.

        Returns:
            List ``[K0, K1]`` where K0 = √(1-p)·I and K1 = √p·Z.
        """
        p = self.p
        K0 = jnp.sqrt(1 - p) * Id._matrix
        K1 = jnp.sqrt(p) * PauliZ._matrix
        return [K0, K1]


class DepolarizingChannel(KrausChannel):
    r"""Single-qubit depolarizing channel.

    .. math::
        K_0 = \sqrt{1-p}\,I,\quad K_1 = \sqrt{p/3}\,X,\quad
        K_2 = \sqrt{p/3}\,Y,\quad K_3 = \sqrt{p/3}\,Z

    where *p* ∈ [0, 1].  At p = 3/4 the channel is fully depolarizing.
    """

    def __init__(self, p: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a depolarizing channel.

        Args:
            p: Depolarization probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *p* is outside [0, 1].
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the four Kraus operators for the depolarizing channel.

        Returns:
            List ``[K0, K1, K2, K3]`` corresponding to I, X, Y, Z components.
        """
        p = self.p
        K0 = jnp.sqrt(1 - p) * Id._matrix
        K1 = jnp.sqrt(p / 3) * PauliX._matrix
        K2 = jnp.sqrt(p / 3) * PauliY._matrix
        K3 = jnp.sqrt(p / 3) * PauliZ._matrix
        return [K0, K1, K2, K3]


class AmplitudeDamping(KrausChannel):
    r"""Single-qubit amplitude damping channel.

    .. math::
        K_0 = \begin{pmatrix}1 & 0\\ 0 & \sqrt{1-\gamma}\end{pmatrix},\quad
        K_1 = \begin{pmatrix}0 & \sqrt{\gamma}\\ 0 & 0\end{pmatrix}

    where *γ* ∈ [0, 1] is the probability of energy loss (|1⟩ → |0⟩).
    """

    def __init__(self, gamma: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an amplitude damping channel.

        Args:
            gamma: Energy-loss probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *gamma* is outside [0, 1].
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.gamma = gamma
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the amplitude damping channel.

        Returns:
            List ``[K0, K1]`` as defined in the class docstring.
        """
        g = self.gamma
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=jnp.complex128)
        K1 = jnp.array([[0.0, jnp.sqrt(g)], [0.0, 0.0]], dtype=jnp.complex128)
        return [K0, K1]


class PhaseDamping(KrausChannel):
    r"""Single-qubit phase damping (dephasing) channel.

    .. math::
        K_0 = \begin{pmatrix}1 & 0\\ 0 & \sqrt{1-\gamma}\end{pmatrix},\quad
        K_1 = \begin{pmatrix}0 & 0\\ 0 & \sqrt{\gamma}\end{pmatrix}

    where *γ* ∈ [0, 1] is the phase damping probability.
    """

    def __init__(self, gamma: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a phase damping channel.

        Args:
            gamma: Phase-damping probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *gamma* is outside [0, 1].
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        self.gamma = gamma
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the phase damping channel.

        Returns:
            List ``[K0, K1]`` as defined in the class docstring.
        """
        g = self.gamma
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=jnp.complex128)
        K1 = jnp.array([[0.0, 0.0], [0.0, jnp.sqrt(g)]], dtype=jnp.complex128)
        return [K0, K1]


class ThermalRelaxationError(KrausChannel):
    r"""Single-qubit thermal relaxation error channel.

    Models simultaneous T₁ energy relaxation and T₂ dephasing.  Two regimes
    are handled:

    **T₂ ≤ T₁** (Markovian dephasing + reset):
        Six Kraus operators built from p_z (phase-flip probability), p_r0
        (reset-to-|0⟩ probability) and p_r1 (reset-to-|1⟩ probability).

    **T₂ > T₁** (non-Markovian; Choi matrix decomposition):
        The Choi matrix is assembled from the relaxation/dephasing rates, then
        diagonalised; Kraus operators are K_i = √λ_i · mat(v_i).

    Attributes:
        pe: Excited-state population (thermal population of |1⟩).
        t1: T₁ longitudinal relaxation time.
        t2: T₂ transverse dephasing time.
        tg: Gate duration.
    """

    def __init__(
        self,
        pe: float,
        t1: float,
        t2: float,
        tg: float,
        wires: Union[int, List[int]] = 0,
    ) -> None:
        """Initialise a thermal relaxation error channel.

        Args:
            pe: Excited-state population (thermal population of |1⟩), in [0, 1].
            t1: T₁ longitudinal relaxation time, must be > 0.
            t2: T₂ transverse dephasing time, must be > 0 and ≤ 2·T₁.
            tg: Gate duration, must be ≥ 0.
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If any parameter violates the stated constraints.
        """
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
        """Return the Kraus operators for the thermal relaxation channel.

        The number of operators depends on the regime:

        * **T₂ ≤ T₁**: six operators (identity, phase-flip, two reset-to-|0⟩,
          two reset-to-|1⟩).
        * **T₂ > T₁**: four operators derived from the Choi matrix eigendecomposition.

        Returns:
            List of 2x2 JAX arrays representing the Kraus operators.
        """
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
            # Each eigenvector (column of eigenvectors) reshaped as 2x2 → one Kraus op
            kraus = []
            for i in range(4):
                lam = eigenvalues[i]
                vec = eigenvectors[:, i]
                mat = jnp.sqrt(jnp.abs(lam)) * vec.reshape(2, 2, order="F")
                kraus.append(mat.astype(jnp.complex128))
            return kraus


class QubitChannel(KrausChannel):
    """Generic Kraus channel from a user-supplied list of Kraus operators.

    This replaces PennyLane's ``qml.QubitChannel`` and accepts an arbitrary set
    of Kraus matrices satisfying Σ_k K_k†K_k = I.

    Example::

        kraus_ops = [jnp.sqrt(0.9) * jnp.eye(2), jnp.sqrt(0.1) * PauliX._matrix]
        QubitChannel(kraus_ops, wires=0)
    """

    def __init__(
        self, kraus_ops: List[jnp.ndarray], wires: Union[int, List[int]] = 0
    ) -> None:
        """Initialise a generic Kraus channel.

        Args:
            kraus_ops: List of Kraus matrices.  Each must be a square 2D array
                of dimension ``2**k x 2**k`` where *k* = ``len(wires)``.
            wires: Qubit index or list of qubit indices this channel acts on.
        """
        self._kraus_ops = [jnp.asarray(K, dtype=jnp.complex128) for K in kraus_ops]
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the stored Kraus operators.

        Returns:
            List of Kraus operator matrices.
        """
        return self._kraus_ops


# ===================================================================
# Pauli algebra helpers
# TODO: this needs refactoring and can be potentially merged into the
# codebase above
# ===================================================================

# Single-qubit Pauli matrices (plain arrays, no Operation overhead)
_PAULI_MATS = [Id._matrix, PauliX._matrix, PauliY._matrix, PauliZ._matrix]
_PAULI_LABELS = ["I", "X", "Y", "Z"]
_PAULI_CLASSES = [Id, PauliX, PauliY, PauliZ]


def adjoint_matrix(op: Operation) -> jnp.ndarray:
    """Return the adjoint (conjugate transpose) of *op*'s matrix.

    Args:
        op: A quantum operation with a ``.matrix`` property.

    Returns:
        The ``(d, d)`` adjoint matrix.
    """
    return jnp.conj(op.matrix).T


def evolve_pauli_with_clifford(
    clifford: Operation,
    pauli: Operation,
    adjoint_left: bool = True,
) -> Operation:
    """Compute C† P C  (or  C P C†)  and return the result as an Operation.

    Both operators are first embedded into the full Hilbert space spanned by
    the union of their wire sets.  The result is wrapped in a
    :class:`Hermitian` so it can be used in further algebra.

    Args:
        clifford: A Clifford gate.
        pauli: A Pauli / Hermitian operator.
        adjoint_left: If ``True``, compute C† P C; otherwise C P C†.

    Returns:
        A :class:`Hermitian` wrapping the evolved matrix.
    """
    all_wires = sorted(set(clifford.wires) | set(pauli.wires))
    n = len(all_wires)

    C = _embed_matrix(clifford.matrix, clifford.wires, all_wires, n)
    P = _embed_matrix(pauli.matrix, pauli.wires, all_wires, n)
    Cd = jnp.conj(C).T

    if adjoint_left:
        result = Cd @ P @ C
    else:
        result = C @ P @ Cd

    return Hermitian(matrix=result, wires=all_wires, record=False)


def _embed_matrix(
    mat: jnp.ndarray,
    op_wires: list,
    all_wires: list,
    n_total: int,
) -> jnp.ndarray:
    """Embed a gate matrix into a larger Hilbert space via tensor products.

    If the gate already acts on all wires, the matrix is returned as-is.
    Otherwise the gate matrix is tensored with identities on the missing
    wires, and the resulting matrix rows/columns are permuted so that qubit
    ordering matches *all_wires*.

    Args:
        mat: The gate's unitary matrix of shape ``(2**k, 2**k)`` where
            ``k = len(op_wires)``.
        op_wires: The wires the gate acts on.
        all_wires: The full ordered list of wires.
        n_total: ``len(all_wires)``.

    Returns:
        A ``(2**n_total, 2**n_total)`` matrix.
    """
    k = len(op_wires)
    if k == n_total and list(op_wires) == list(all_wires):
        return mat

    # Build the full-space matrix by tensoring with identities
    # Strategy: tensor I on missing wires, then permute
    missing = [w for w in all_wires if w not in op_wires]
    # Full matrix = mat ⊗ I_{missing}
    full_mat = mat
    for _ in missing:
        full_mat = jnp.kron(full_mat, jnp.eye(2, dtype=jnp.complex128))

    # The current ordering is [op_wires..., missing...]
    # We need to permute to match all_wires ordering
    current_order = list(op_wires) + missing
    if current_order != list(all_wires):
        perm = [current_order.index(w) for w in all_wires]
        full_mat = _permute_matrix(full_mat, perm, n_total)

    return full_mat


def _permute_matrix(mat: jnp.ndarray, perm: list, n_qubits: int) -> jnp.ndarray:
    """Permute the qubit ordering of a matrix.

    Given a ``(2**n, 2**n)`` matrix and a permutation of ``[0..n-1]``,
    reorder the qubits so that qubit ``i`` moves to position ``perm[i]``.

    Args:
        mat: Square matrix of dimension ``2**n_qubits``.
        perm: Permutation list.
        n_qubits: Number of qubits.

    Returns:
        Permuted matrix of the same shape.
    """
    dim = 2**n_qubits
    # Reshape to tensor, permute axes, reshape back
    tensor = mat.reshape([2] * (2 * n_qubits))
    # Axes: first n_qubits are row indices, last n_qubits are column indices
    row_perm = perm
    col_perm = [p + n_qubits for p in perm]
    tensor = jnp.transpose(tensor, row_perm + col_perm)
    return tensor.reshape(dim, dim)


def pauli_decompose(matrix: jnp.ndarray, wire_order: Optional[List[int]] = None):
    r"""Decompose a Hermitian matrix into a sum of Pauli tensor products.

    For an *n*-qubit matrix (``2**n x 2**n``), returns the dominant Pauli
    term (the one with the largest absolute coefficient), wrapped as an
    :class:`Operation`.  This is sufficient for the Fourier-tree algorithm
    which only needs the single non-zero Pauli term produced by Clifford
    conjugation of a Pauli operator.

    The decomposition uses the trace formula:
    ``c_P = Tr(P · M) / 2**n``

    Args:
        matrix: A ``(2**n, 2**n)`` Hermitian matrix.
        wire_order: Optional list of wire indices.  If ``None``, defaults
            to ``[0, 1, ..., n-1]``.

    Returns:
        A tuple ``(coeff, op)`` where *coeff* is the complex coefficient and
        *op* is the Pauli :class:`Operation` (PauliX, PauliY, PauliZ, I, or
        a :class:`Hermitian` for multi-qubit tensor products).
    """
    from itertools import product as _product
    from functools import reduce as _reduce

    dim = matrix.shape[0]
    n_qubits = int(jnp.log2(dim))

    if wire_order is None:
        wire_order = list(range(n_qubits))

    # For single qubit, fast path
    if n_qubits == 1:
        best_idx, best_coeff = 0, 0.0
        for idx, P in enumerate(_PAULI_MATS):
            coeff = jnp.trace(P @ matrix) / 2.0
            if jnp.abs(coeff) > jnp.abs(best_coeff):
                best_idx = idx
                best_coeff = coeff
        op_cls = _PAULI_CLASSES[best_idx]
        result_op = op_cls(wires=wire_order[0], record=False)
        result_op._pauli_label = _PAULI_LABELS[best_idx]
        return best_coeff, result_op

    # Multi-qubit: iterate over all Pauli tensor products
    best_label = None
    best_coeff = 0.0
    for indices in _product(range(4), repeat=n_qubits):
        P = _reduce(jnp.kron, [_PAULI_MATS[i] for i in indices])
        coeff = jnp.trace(P @ matrix) / dim
        if jnp.abs(coeff) > jnp.abs(best_coeff):
            best_coeff = coeff
            best_label = indices

    # Build the Pauli string label
    pauli_label = "".join(_PAULI_LABELS[i] for i in best_label)

    # Build the operation for the dominant term
    if sum(1 for i in best_label if i != 0) <= 1:
        # Single-qubit Pauli on one wire
        for q, idx in enumerate(best_label):
            if idx != 0:
                op_cls = _PAULI_CLASSES[idx]
                result_op = op_cls(wires=wire_order[q], record=False)
                result_op._pauli_label = _PAULI_LABELS[idx]
                return best_coeff, result_op
        # All identity
        result_op = Id(wires=wire_order[0], record=False)
        result_op._pauli_label = "I" * n_qubits
        return best_coeff, result_op
    else:
        # Multi-qubit tensor product → Hermitian with pauli label attached
        P = _reduce(jnp.kron, [_PAULI_MATS[i] for i in best_label])
        result_op = Hermitian(matrix=P, wires=wire_order, record=False)
        result_op._pauli_label = pauli_label
        return best_coeff, result_op


def pauli_string_from_operation(op: Operation) -> str:
    """Extract a Pauli word string from an operation.

    Maps ``PauliX`` → ``"X"``, ``PauliY`` → ``"Y"``, ``PauliZ`` → ``"Z"``,
    ``I`` → ``"I"``.  For :class:`PauliRot`, returns its stored ``pauli_word``.
    For operations produced by :func:`pauli_decompose`, returns the stored
    ``_pauli_label`` attribute.

    Args:
        op: A quantum operation.

    Returns:
        A string like ``"X"``, ``"ZZ"``, etc.
    """
    if isinstance(op, PauliRot) and hasattr(op, "pauli_word"):
        return op.pauli_word
    # Check for label stored by pauli_decompose
    if hasattr(op, "_pauli_label"):
        return op._pauli_label
    name_map = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "I": "I"}
    if op.name in name_map:
        return name_map[op.name]
    # Fall back: decompose the matrix
    _, pauli_op = pauli_decompose(op.matrix, wire_order=op.wires)
    return pauli_op._pauli_label
