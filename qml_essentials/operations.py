from typing import Callable, List, Optional, Tuple, Union
from functools import lru_cache
import string
import numpy as np

import jax
import jax.numpy as jnp

from qml_essentials.tape import active_tape, recording  # noqa: F401 (re-export)


def _cdtype():
    """Return the active JAX complex dtype
    (complex128 if x64 enabled, else complex64).
    """
    return jnp.complex128 if jax.config.x64_enabled else jnp.complex64


@lru_cache(maxsize=256)
def _einsum_subscript(
    n: int,
    k: int,
    target_axes: Tuple[int, ...],
) -> str:
    """Build an ``einsum`` subscript that fuses contraction + axis restore.

    Args:
        n: Total rank of the state tensor (number of qubits for statevectors,
            ``2 * n_qubits`` for density matrices).
        k: Number of qubits the gate acts on.
        target_axes: Tuple of k axis indices in the state tensor that the
            gate contracts against.

    Returns:
        ``einsum`` subscript string, e.g. ``"ab,cBd->cad"`` for a 1-qubit
        gate on wire 1 of a 3-qubit state.
    """
    letters = string.ascii_letters
    # State indices: one letter per axis
    state_idx = list(letters[:n])
    # Contracted indices (the ones being replaced by the gate)
    contracted = [state_idx[ax] for ax in target_axes]
    # Gate indices: new output indices + contracted input indices
    new_out = [letters[n + i] for i in range(k)]  # fresh letters for output
    gate_idx = new_out + contracted  # gate shape: (out0, out1, ..., in0, in1, ...)
    # Result indices: replace target axes with new output letters
    result_idx = list(state_idx)
    for i, ax in enumerate(target_axes):
        result_idx[ax] = new_out[i]
    return "".join(gate_idx) + "," + "".join(state_idx) + "->" + "".join(result_idx)


def _contract_and_restore(
    tensor: jnp.ndarray,
    gate: jnp.ndarray,
    k: int,
    target_axes: List[int],
) -> jnp.ndarray:
    """Contract gate against target_axes of tensor and restore axis order.

    The einsum subscript is cached via :func:`_einsum_subscript` so the
    string construction only happens once per unique
    ``(total, k, target_axes)`` combination.

    Args:
        tensor: Rank-N tensor (e.g. ``(2,)*n`` for states or ``(2,)*2n``
            for density matrices).
        gate: Reshaped gate tensor of shape ``(2,)*2k``.
        k: Number of qubits the gate acts on (= ``len(target_axes)``).
        target_axes: The k axes of tensor to contract against.

    Returns:
        Updated tensor with the same rank as tensor, with the
        contracted axes restored to their original positions.
    """
    subscript = _einsum_subscript(tensor.ndim, k, tuple(target_axes))
    return jnp.einsum(subscript, gate, tensor)


class Operation:
    """Base class for any quantum operation or observable.

    Further gates should inherit from this class to realise more specific
    operations.  Generally, operations are created by instantiation inside a
    circuit function passed to :class:`Script`; the instance is
    automatically appended to the active tape.

    An ``Operation`` can also serve as an *observable*: its matrix is used to
    compute expectation values via ``apply_to_state`` / ``apply_to_density``.

    Attributes:
        _matrix: Class-level default gate matrix.  Subclasses set this to their
            fixed unitary.  Instances may override it via the *matrix* argument
            to ``__init__``.
        _num_wires: Expected number of wires for this gate.  Subclasses set
            this to enforce wire count validation.  ``None`` means any number
            of wires is accepted.
        _param_names: Tuple of attribute names for the gate parameters.
            Used by :attr:`parameters` and :meth:`__repr__`.
    """

    # Subclasses should set this to the gate's unitary / matrix
    _matrix: jnp.ndarray = None
    is_controlled = False
    _num_wires: Optional[int] = None
    _param_names: Tuple[str, ...] = ()

    def __init__(
        self,
        wires: Union[int, List[int]] = 0,
        matrix: Optional[jnp.ndarray] = None,
        record: bool = True,
        input_idx: int = -1,
        name: Optional[str] = None,
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
            input_idx: Marks the operation as input with the corresponding
                input index, which is useful for the analytical Fourier
                coefficients computation, but has no effect otherwise.
            name: Optional explicit name for this operation.  When ``None``
                (default), the class name is used (e.g. ``"RX"``).

        Raises:
            ValueError: If ``_num_wires`` is set and the number of wires
                doesn't match, or if duplicate wires are provided.
        """
        self.name = name or self.__class__.__name__
        self.wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]
        self.input_idx = input_idx

        if self._num_wires is not None and len(self.wires) != self._num_wires:
            raise ValueError(
                f"{self.name} expects {self._num_wires} wire(s), "
                f"got {len(self.wires)}: {self.wires}"
            )
        if len(self.wires) != len(set(self.wires)):
            raise ValueError(f"{self.name} received duplicate wires: {self.wires}")

        if matrix is not None:
            self._matrix = matrix

        # If a tape is currently recording, append ourselves
        if record:
            tape = active_tape()
            if tape is not None:
                tape.append(self)

    @property
    def parameters(self) -> list:
        """Return the list of numeric parameters for this operation.

        Uses the declarative ``_param_names`` tuple to collect parameter
        values in a canonical order.  Non-parametrized gates return an
        empty list.

        Returns:
            List of parameter values (floats or JAX arrays).
        """
        return [getattr(self, name) for name in self._param_names]

    def __repr__(self) -> str:
        """Return a human-readable representation of this operation.

        Returns:
            A string like ``"RX(0.5000, wires=[0])"`` or ``"CX(wires=[0, 1])"``.
        """
        params = self.parameters
        if params:
            param_str = ", ".join(
                (
                    f"{float(v):.4f}"
                    if isinstance(v, (float, np.floating, jnp.ndarray))
                    else str(v)
                )
                for v in params
            )
            return f"{self.name}({param_str}, wires={self.wires})"
        return f"{self.name}(wires={self.wires})"

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

    @property
    def input_idx(self) -> int:
        """The index of an input

        Returns:
            input_idx: Index of the input
        """
        return self._input_idx

    @input_idx.setter
    def input_idx(self, input_idx: int) -> None:
        """Setter for the input_idx flag

        Args:
            input_idx: Index of the input
        """
        self._input_idx = input_idx

    def _update_tape_operation(self, op: "Operation") -> None:
        """
        If ``self`` is already on the active tape (the typical case when
        chaining ``Gate(...).dagger()``), it is replaced by the daggered
        operation so that only U\\dagger appears on the tape —
        not both U and ``U\\dagger``.
        Note that this should only be called immediately after the tape is updated.s

        Args:
            op (Operation): New replaced operation on the tape
        """
        # If self was recorded on the tape, replace it with the daggered op.
        tape = active_tape()
        if tape is not None:
            if tape and tape[-1] is self:
                tape[-1] = op
            else:
                tape.append(op)

    def dagger(self) -> "Operation":
        """Return a new operation, the conjugate transpose (``U\\dagger``)
        Usage inside a circuit function::

            RX(0.5, wires=0).dagger()

        Returns:
            A new :class:`Operation` with matrix ``U\\dagger`` acting on the same wires.
        """
        mat = jnp.conj(self._matrix).T
        op = Operation(wires=self.wires, matrix=mat, record=False)

        self._update_tape_operation(op)

        return op

    def power(self, power) -> "Operation":
        """Return a new operation, the power (``U^power``)
        Usage inside a circuit function::

            PauliX(wires=0).power(2)

        Returns:
            A new :class:`Operation` with matrix ``U\\dagger`` acting on the same wires.
        """
        # TODO: support fractional powers
        mat = jnp.linalg.matrix_power(self._matrix, power)
        op = Operation(wires=self.wires, matrix=mat, record=False)

        self._update_tape_operation(op)

        return op

    def __mul__(self, factor: float) -> "Operation":
        """Return a new operation, the product between U and a scalar (``U*x``)
        Usage inside a circuit function::

            PauliX(wires=0) * x

        Returns:
            A new :class:`Operation` with matrix ``U*x`` acting on the same wires.
        """
        mat = factor * self._matrix
        op = Operation(wires=self.wires, matrix=mat, record=False)

        self._update_tape_operation(op)

        return op

    # Also overwrite * for right operands
    __rmul__ = __mul__

    def __add__(self, other: "Operation") -> "Operation":
        """Element-wise addition of two operations on the same wires.

        Returns:
            A new :class:`Operation` whose matrix is the sum of both matrices.

        Raises:
            ValueError: If the wire sets differ.
        """
        if sorted(self.wires) != sorted(other.wires):
            raise ValueError(
                f"Can only add operations acting on the same set of wires, "
                f"got {self.wires} and {other.wires}"
            )

        op = Operation(
            wires=self.wires,
            matrix=self.matrix + other.matrix,
            record=False,
        )
        return op

    def __matmul__(self, other: "Operation") -> "Operation":
        """Tensor (Kronecker) product of two operations.

        The resulting operation acts on the union of both wire sets and
        carries the Kronecker product of both matrices.  Wire sets must
        be disjoint.

        Returns:
            A new :class:`Operation` whose matrix is
            ``self.matrix \\otimes other.matrix``
            and whose wires are the concatenation of both wire lists.

        Raises:
            ValueError: If the two operations share any wires.
        """
        if set(self.wires) & set(other.wires):
            raise ValueError(
                f"Cannot take tensor product: overlapping wires "
                f"{self.wires} and {other.wires}"
            )
        new_matrix = jnp.kron(self.matrix, other.matrix)
        new_wires = self.wires + other.wires
        op = Operation(wires=new_wires, matrix=new_matrix, record=False)
        return op

    def lifted_matrix(self, n_qubits: int) -> jnp.ndarray:
        """Return the full ``2**n x 2**n`` matrix embedding this gate.

        Embeds the ``k``-qubit gate matrix into the ``n``-qubit Hilbert space
        by applying it to the identity matrix via :meth:`apply_to_state`.
        This is useful for computing ``Tr(O·\\rho )`` directly without vmap.

        Args:
            n_qubits: Total number of qubits in the circuit.

        Returns:
            The ``(2**n, 2**n)`` matrix of this operation in the full space.
        """
        dim = 2**n_qubits
        # Apply the gate to each basis vector (column of identity)
        return jax.vmap(lambda col: self.apply_to_state(col, n_qubits))(
            jnp.eye(dim, dtype=_cdtype())
        ).T

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a statevector via tensor contraction.

        The statevector (shape ``(2**n,)``) is reshaped into a rank-n tensor
        of shape ``(2,)*n``.  The gate (shape ``(2**k, 2**k)``) is reshaped to
        ``(2,)*2k`` and contracted against the k target wire axes.

        Memory footprint is O(2**n) and the operation supports arbitrary k.
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

    def apply_to_state_tensor(self, psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a statevector already in tensor form.

        Like :meth:`apply_to_state` but expects the state in rank-n tensor
        form ``(2,)*n`` and returns the result in the same form.  This avoids
        the ``reshape`` calls at the per-gate level when the simulation loop
        keeps the state in tensor form throughout.

        Args:
            psi: Statevector tensor of shape ``(2,)*n_qubits``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated statevector tensor of shape ``(2,)*n_qubits``.
        """
        k = len(self.wires)
        gate_tensor = self._gate_tensor(k)
        return _contract_and_restore(psi, gate_tensor, k, self.wires)

    def _gate_tensor(self, k: int) -> jnp.ndarray:
        """Return the gate matrix reshaped to ``(2,)*2k`` tensor form.

        The result is cached on the instance so repeated calls (e.g. from
        density-matrix simulation which applies U and U*) avoid redundant
        reshape dispatch.

        Args:
            k: Number of qubits the gate acts on.

        Returns:
            Gate matrix as a rank-2k tensor of shape ``(2,)*2k``.
        """
        cached = getattr(self, "_cached_gate_tensor", None)
        if cached is not None:
            return cached
        gt = self.matrix.reshape((2,) * 2 * k)
        # Only cache for non-parametrized gates (whose matrix is a class attr)
        if self._matrix is self.__class__._matrix:
            object.__setattr__(self, "_cached_gate_tensor", gt)
        return gt

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a density matrix via \\rho -> U\\rho U\\dagger.

        The density matrix (shape ``(2**n, 2**n)``) is treated as a rank-*2n*
        tensor with n "ket" axes (0..n-1) and n "bra" axes (n..2n-1).
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
        U = self._gate_tensor(k)
        U_conj = jnp.conj(U)

        rho_t = rho.reshape((2,) * 2 * n_qubits)

        # Apply U to ket axes, U\\dagger to bra axes
        rho_t = _contract_and_restore(rho_t, U, k, self.wires)
        bra_wires = [w + n_qubits for w in self.wires]
        rho_t = _contract_and_restore(rho_t, U_conj, k, bra_wires)

        return rho_t.reshape(2**n_qubits, 2**n_qubits)


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
            matrix=jnp.asarray(matrix, dtype=_cdtype()),
            record=record,
        )

    def __rmul__(self, coeff_fn: Callable) -> "ParametrizedHamiltonian":
        """Support ``coeff_fn * Hermitian`` -> :class:`ParametrizedHamiltonian`.

        Args:
            coeff_fn (Callable): A callable ``(params, t) -> scalar`` giving the
                time-dependent coefficient.

        Returns:
            ParametrizedHamiltonian: A :class:`ParametrizedHamiltonian` pairing
                *coeff_fn* with this operator's matrix and wires.

        Raises:
            TypeError: If *coeff_fn* is not callable.
        """
        if not callable(coeff_fn):
            raise TypeError(
                f"Left operand of `* Hermitian` must be callable, got {type(coeff_fn)}"
            )
        return ParametrizedHamiltonian(terms=[(coeff_fn, self.matrix, self.wires)])


class ParametrizedHamiltonian:
    """A time-dependent Hamiltonian as a sum of ``coeff * Hermitian`` terms.

    Mathematically::

        H(t) = \\sum_i f_i(params_i, t) * H_i

    Construction is always done from an explicit list of
    ``(coeff_fn, H_mat, wires)`` triples passed as ``terms``.  The
    common single-term shorthand is the operator form
    ``coeff_fn * Hermitian(matrix, wires)`` (see
    :meth:`Hermitian.__rmul__`), which returns a one-term instance.
    Multi-term Hamiltonians are composed with ``+`` between
    :class:`ParametrizedHamiltonian` instances::

        H1 = coeff_x * Hermitian(X, wires=0)
        H2 = coeff_y * Hermitian(Y, wires=0)
        H_td = H1 + H2

        # evolve under the composite Hamiltonian; coeff_args is a list of
        # parameter sets, one per term, in the order the terms were added:
        evolve(H_td)([px, py], T=1.0)

    Attributes:
        coeff_fns: Tuple of callables ``(params, t) -> scalar``, one per term.
        H_mats: Tuple of static Hermitian matrices, one per term.
        wires: Wires this Hamiltonian acts on (union across all terms; for
            now all terms are required to share the same wire set).
    """

    def __init__(
        self,
        terms: List[Tuple[Callable, jnp.ndarray, Union[int, List[int]]]],
    ) -> None:
        """Build a (possibly multi-term) parametrized Hamiltonian.

        Args:
            terms: List of ``(coeff_fn, H_mat, wires)`` triples.  Use the
                ``coeff_fn * Hermitian(...)`` shorthand to build a
                one-term instance; combine instances with ``+`` to add
                terms.

        Raises:
            ValueError: If the term list is empty, or if terms act on
                differing wire sets (multi-wire broadcasting is
                deferred — see :mod:`yaqsi`), or if term matrices have
                incompatible shapes.
        """
        if len(terms) == 0:
            raise ValueError("ParametrizedHamiltonian needs at least one term.")

        # Normalise wires (single int -> [int]) and validate consistency.
        def _wlist(w):
            return [w] if isinstance(w, int) else list(w)

        first_wires = _wlist(terms[0][2])
        for _, _, w in terms[1:]:
            if _wlist(w) != first_wires:
                raise ValueError(
                    "All terms of a ParametrizedHamiltonian must currently "
                    "act on the same wires; got "
                    f"{_wlist(w)} vs. {first_wires}. "
                    "Multi-wire broadcasting across terms is not yet supported."
                )

        # Validate matrix shape compatibility across terms.
        first_dim = jnp.asarray(terms[0][1]).shape
        for _, H, _ in terms[1:]:
            if jnp.asarray(H).shape != first_dim:
                raise ValueError(
                    f"All term matrices must have the same shape; got "
                    f"{jnp.asarray(H).shape} vs. {first_dim}."
                )

        self._terms: Tuple[Tuple[Callable, jnp.ndarray, List[int]], ...] = tuple(
            (fn, jnp.asarray(H, dtype=_cdtype()), _wlist(w)) for fn, H, w in terms
        )
        self.wires: List[int] = list(first_wires)

    # --- term accessors -------------------------------------------------

    @property
    def coeff_fns(self) -> Tuple[Callable, ...]:
        """Tuple of coefficient functions, one per term."""
        return tuple(fn for fn, _, _ in self._terms)

    @property
    def H_mats(self) -> Tuple[jnp.ndarray, ...]:
        """Tuple of Hermitian matrices, one per term."""
        return tuple(H for _, H, _ in self._terms)

    @property
    def n_terms(self) -> int:
        """Number of terms in the Hamiltonian."""
        return len(self._terms)

    # --- composition ---------------------------------------------------

    def __add__(self, other: "ParametrizedHamiltonian") -> "ParametrizedHamiltonian":
        """Concatenate term lists: ``H = H1 + H2``."""
        if not isinstance(other, ParametrizedHamiltonian):
            return NotImplemented
        return ParametrizedHamiltonian(terms=list(self._terms) + list(other._terms))

    def __neg__(self) -> "ParametrizedHamiltonian":
        """Negate every coefficient: ``-H`` = sum of ``(-f_i) * H_i``."""
        new_terms = [
            ((lambda f: lambda p, t: -f(p, t))(fn), H, w) for fn, H, w in self._terms
        ]
        return ParametrizedHamiltonian(terms=new_terms)

    def __sub__(self, other: "ParametrizedHamiltonian") -> "ParametrizedHamiltonian":
        if not isinstance(other, ParametrizedHamiltonian):
            return NotImplemented
        return self + (-other)


class Id(Operation):
    """Identity gate.

    Supports an arbitrary number of wires.  When more than one wire is
    given the matrix is the ``2**k x 2**k`` identity (where *k* is the
    number of wires).
    """

    _matrix = jnp.eye(2, dtype=_cdtype())
    _num_wires = None  # accept any number of wires

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise an identity gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
                When multiple wires are given the matrix is automatically
                expanded to the matching ``2**k × 2**k`` identity.
        """
        w = list(wires) if isinstance(wires, (list, tuple)) else [wires]
        k = len(w)
        if k > 1:
            kwargs["matrix"] = jnp.eye(2**k, dtype=_cdtype())
        super().__init__(wires=wires, **kwargs)


class PauliX(Operation):
    """Pauli-X gate / observable (bit-flip, \\sigma_x)."""

    _matrix = jnp.array([[0, 1], [1, 0]], dtype=_cdtype())
    _num_wires = 1

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-X gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class PauliY(Operation):
    """Pauli-Y gate / observable (\\sigma_y)."""

    _matrix = jnp.array([[0, -1j], [1j, 0]], dtype=_cdtype())
    _num_wires = 1

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-Y gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class PauliZ(Operation):
    """Pauli-Z gate / observable (phase-flip, \\sigma_z)."""

    _matrix = jnp.array([[1, 0], [0, -1]], dtype=_cdtype())
    _num_wires = 1

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-Z gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class H(Operation):
    """Hadamard gate."""

    _matrix = jnp.array([[1, 1], [1, -1]], dtype=_cdtype()) / jnp.sqrt(2)
    _num_wires = 1

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Hadamard gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class S(Operation):
    """S (phase) gate — a Clifford gate equal to \\sqrt Z.

    .. math::
        S = \\begin{pmatrix}1 & 0\\ 0 & i\\end{pmatrix}
    """

    _matrix = jnp.array([[1, 0], [0, 1j]], dtype=_cdtype())
    _num_wires = 1

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an S gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class SWAP(Operation):
    """SWAP gate."""

    _matrix = jnp.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=_cdtype()
    )
    _num_wires = 2

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a SWAP gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class RandomUnitary(Operation):
    """Creates a random hermitian matrix and applies it as a gate."""

    def __init__(
        self,
        wires: Union[int, List[int]],
        key: jax.random.PRNGKey,
        scale: float = 1.0,
        record: bool = True,
    ) -> None:
        """Initialise a random unitary gate.

        Args:
            wires (Union[int, List[int]]): Qubit index or list of qubit indices
                this gate acts on.
            key (jax.random.PRNGKey): PRNGKey for randomization.
            scale (float): Scale of the random unitary (default: 1.0).
            record (bool): Whether to record this gate on the active tape.
        """
        dim = 2 ** len(wires)
        key_a, key_b = jax.random.split(key)

        A = (
            jax.random.normal(key=key_a, shape=(dim, dim))
            + 1j * jax.random.normal(key=key_b, shape=(dim, dim))
        ).astype(_cdtype())
        H = (A + A.conj().T) / 2.0

        H *= scale / jnp.linalg.norm(H, ord="fro")

        super().__init__(wires, matrix=H, record=record)


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

    def apply_to_state_tensor(self, psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """No-op: return the state tensor unchanged."""
        return psi

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """No-op: return the density matrix unchanged."""
        return rho


def _make_rotation_gate(pauli_class: type, name: str) -> type:
    """Factory for single-qubit rotation gates RX, RY, RZ.

    Each gate has the form ``R_P(\\theta) = cos(\\theta/2) I - i sin(\\theta/2) P``.

    Args:
        pauli_class: One of PauliX, PauliY, PauliZ.
        name: Class name for the generated gate (e.g. ``"RX"``).

    Returns:
        A new :class:`Operation` subclass.
    """
    pauli_mat = pauli_class._matrix

    class _RotationGate(Operation):
        # Fancy way of setting docstring to make it generic
        __doc__ = (
            f"Rotation around the {name[1]} axis: {name}(\\theta) =\n"
            f"exp(-i \\theta/2 {name[1]}).\n"
        )
        _num_wires = 1
        _param_names = ("theta",)

        def __init__(
            self, theta: float, wires: Union[int, List[int]] = 0, **kwargs
        ) -> None:
            self.theta = theta
            c = jnp.cos(theta / 2)
            s = jnp.sin(theta / 2)
            mat = c * Id._matrix - 1j * s * pauli_mat
            super().__init__(wires=wires, matrix=mat, **kwargs)

        def generator(self) -> Operation:
            """Return the generator as the corresponding Pauli operation."""
            return pauli_class(wires=self.wires[0], record=False)

    _RotationGate.__name__ = name
    _RotationGate.__qualname__ = name
    return _RotationGate


RX = _make_rotation_gate(PauliX, "RX")
RY = _make_rotation_gate(PauliY, "RY")
RZ = _make_rotation_gate(PauliZ, "RZ")


# Projectors used by controlled-gate factories
_P0 = jnp.array([[1, 0], [0, 0]], dtype=_cdtype())
_P1 = jnp.array([[0, 0], [0, 1]], dtype=_cdtype())


def _make_controlled_gate(target_class: type, name: str) -> type:
    """Factory for controlled Pauli gates CX, CY, CZ.

    Each gate has the form
    ``CP = |0><0| \\otimes I + |1\\langle\\rangle 1| \\otimes P``.

    Args:
        target_class: The single-qubit gate class (PauliX, PauliY, PauliZ).
        name: Class name for the generated gate (e.g. ``"CX"``).

    Returns:
        A new :class:`Operation` subclass.
    """
    target_mat = target_class._matrix

    class _ControlledGate(Operation):
        __doc__ = (
            f"Controlled-{target_class.__name__[5:]} gate.\n\n"
            f"Applies {target_class.__name__} on the target qubit conditioned "
            f"on the control qubit being in state |1\\rangle."
        )
        _matrix = jnp.kron(_P0, Id._matrix) + jnp.kron(_P1, target_mat)
        _num_wires = 2
        is_controlled = True

        def __init__(self, wires: List[int] = [0, 1], **kwargs) -> None:
            super().__init__(wires=wires, **kwargs)

    _ControlledGate.__name__ = name
    _ControlledGate.__qualname__ = name
    return _ControlledGate


CX = _make_controlled_gate(PauliX, "CX")
CY = _make_controlled_gate(PauliY, "CY")
CZ = _make_controlled_gate(PauliZ, "CZ")


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
        dtype=_cdtype(),
    )
    is_controlled = True
    _num_wires = 3

    def __init__(self, wires: List[int] = [0, 1, 2], **kwargs) -> None:
        """Initialise a Toffoli (CCX) gate.

        Args:
            wires: Three-element list ``[control0, control1, target]``.
        """
        super().__init__(wires=wires, **kwargs)


class CSWAP(Operation):
    """Controlled-SWAP (Fredkin) gate.

    Swaps the two target qubits conditioned on the control qubit being |1\\rangle.

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
        dtype=_cdtype(),
    )
    is_controlled = True
    _num_wires = 3

    def __init__(self, wires: List[int] = [0, 1, 2], **kwargs) -> None:
        """Initialise a Controlled-SWAP (Fredkin) gate.

        Args:
            wires: Three-element list ``[control, target0, target1]``.
        """
        super().__init__(wires=wires, **kwargs)


def _make_controlled_rotation_gate(pauli_class: type, name: str) -> type:
    """Factory for controlled rotation gates CRX, CRY, CRZ.

    Each gate has the form
    ``CR_P(\\theta) = |0><0| \\otimes I + |1><1| \\otimes R_P(\\theta)``.

    Args:
        pauli_class: One of PauliX, PauliY, PauliZ.
        name: Class name for the generated gate (e.g. ``"CRX"``).

    Returns:
        A new :class:`Operation` subclass.
    """
    pauli_mat = pauli_class._matrix

    class _CRotationGate(Operation):
        __doc__ = (
            f"Controlled rotation around the {name[2]} axis.\n\n"
            f"Applies R{name[2]}(\\theta) on the target qubit conditioned on the "
            f"control qubit being in state |1\\rangle.\n\n"
            f".. math::\n"
            f"{name}(\\theta) = |0\\rangle\\langle 0| \\otimes I\n"
            f"                  + |1\\rangle\\langle 1| \\otimes R{name[2]}(\\theta)"
        )
        _num_wires = 2
        _param_names = ("theta",)
        is_controlled = True

        def __init__(self, theta: float, wires: List[int] = [0, 1], **kwargs) -> None:
            self.theta = theta
            c = jnp.cos(theta / 2)
            s = jnp.sin(theta / 2)
            rot = c * Id._matrix - 1j * s * pauli_mat
            mat = jnp.kron(_P0, Id._matrix) + jnp.kron(_P1, rot)
            super().__init__(wires=wires, matrix=mat, **kwargs)

    _CRotationGate.__name__ = name
    _CRotationGate.__qualname__ = name
    return _CRotationGate


CRX = _make_controlled_rotation_gate(PauliX, "CRX")
CRY = _make_controlled_rotation_gate(PauliY, "CRY")
CRZ = _make_controlled_rotation_gate(PauliZ, "CRZ")


class ControlledPhaseShift(Operation):
    r"""Controlled phase shift gate (CPhase).

    Applies a phase shift of ``exp(i * phi)`` to the |11⟩ component of the
    two-qubit state, leaving all other computational basis states unchanged.
    This is a generalization of the CZ gate: when ``phi = \\pi`` the gate
    reduces to CZ.

    .. math::
        \text{CPhase}(\phi) = \text{diag}(1, 1, 1, e^{i\phi})

    which is equivalent to
    ``|0⟩⟨0| \\otimes I + |1⟩⟨1| \\otimes P(phi)`` where
    ``P(phi) = diag(1, exp(i*phi))``.
    """

    _num_wires = 2
    _param_names = ("phi",)
    is_controlled = True

    def __init__(self, phi: float, wires: List[int] = [0, 1], **kwargs) -> None:
        """Initialise a controlled phase shift gate.

        Args:
            phi: Phase shift angle in radians.
            wires: Two-element list ``[control, target]``.
        """
        self.phi = phi
        phase_gate = jnp.array([[1, 0], [0, jnp.exp(1j * phi)]], dtype=_cdtype())
        mat = jnp.kron(_P0, Id._matrix) + jnp.kron(_P1, phase_gate)
        super().__init__(wires=wires, matrix=mat, **kwargs)


class Rot(Operation):
    """General single-qubit rotation:
    Rot(\\phi, \\theta, \\omega) = RZ(\\omega) RY(\\theta) RZ(\\phi).

    This is the most general SU(2) rotation (up to a global phase).  It
    decomposes into three successive rotations and has three free parameters.
    """

    _num_wires = 1
    _param_names = ("phi", "theta", "omega")

    def __init__(
        self,
        phi: float,
        theta: float,
        omega: float,
        wires: Union[int, List[int]] = 0,
        **kwargs,
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
        # Rot(\\phi, \theta, \\omega) = RZ(\\omega) @ RY(\theta) @ RZ(\\phi)
        rz_phi = jnp.cos(phi / 2) * Id._matrix - 1j * jnp.sin(phi / 2) * PauliZ._matrix
        ry_theta = (
            jnp.cos(theta / 2) * Id._matrix - 1j * jnp.sin(theta / 2) * PauliY._matrix
        )
        rz_omega = (
            jnp.cos(omega / 2) * Id._matrix - 1j * jnp.sin(omega / 2) * PauliZ._matrix
        )
        mat = rz_omega @ ry_theta @ rz_phi
        super().__init__(wires=wires, matrix=mat, **kwargs)


class PauliRot(Operation):
    """Multi-qubit Pauli rotation: exp(-i \\theta/2 P) for a Pauli word P.

    The Pauli word is given as a string of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
    characters (one per qubit).  The rotation matrix is computed as
    ``cos(\\theta/2) I - i sin(\\theta/2) P`` where *P* is the tensor product of the
    corresponding single-qubit Pauli matrices.

    Example::

        PauliRot(0.5, "XY", wires=[0, 1])
    """

    _param_names = ("theta",)

    # Map from character to 2x2 matrix
    _PAULI_MAP = {
        "I": Id._matrix,
        "X": PauliX._matrix,
        "Y": PauliY._matrix,
        "Z": PauliZ._matrix,
    }

    def __init__(
        self, theta: float, pauli_word: str, wires: Union[int, List[int]] = 0, **kwargs
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
            jnp.cos(theta / 2) * jnp.eye(dim, dtype=_cdtype())
            - 1j * jnp.sin(theta / 2) * P
        )
        super().__init__(wires=wires, matrix=mat, **kwargs)

    def generator(self) -> Operation:
        """Return the generator Pauli tensor product as an :class:`Operation`.

        The generator of ``PauliRot(\\theta, word, wires)`` is the tensor product
        of single-qubit Pauli matrices specified by *word*.  The returned
        :class:`Hermitian` wraps that matrix and the gate's wires.

        Returns:
            :class:`Hermitian` operation representing the Pauli tensor product.
        """
        from functools import reduce as _reduce

        pauli_matrices = [self._PAULI_MAP[c] for c in self.pauli_word]
        P = _reduce(jnp.kron, pauli_matrices)
        return Hermitian(matrix=P, wires=self.wires, record=False)


class KrausChannel(Operation):
    """Base class for noise channels defined by a set of Kraus operators.

    A Kraus channel \\phi(\\rho ) = \\sigma_k K_k \\rho  K_k\\dagger
    is the most general physical
    operation on a quantum state.  For a pure unitary gate there is a single
    operator K_0 = U satisfying K_0\\daggerK_0 = I; for noisy channels there are
    multiple operators.

    Subclasses must implement :meth:`kraus_matrices` and return a list of JAX
    arrays.  :meth:`apply_to_state` is intentionally left unimplemented:
    Kraus channels require a density-matrix representation and cannot be
    applied to a pure statevector in general.
    """

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the list of Kraus operators for this channel.

        Returns:
            List of 2-D JAX arrays, each of shape ``(2**k, 2**k)`` where k
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

    def apply_to_state_tensor(self, psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Raises TypeError — noise channels require density-matrix simulation."""
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and cannot be "
            "applied to a pure statevector. Use execute(type='density') instead."
        )

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply
        \\phi(\\rho ) = \\sigma_k K_k \\rho  K_k\\dagger using tensor-contraction.

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


class BitFlip(KrausChannel):
    r"""Single-qubit bit-flip (Pauli-X) error channel.

    .. math::
        K_0 = \sqrt{1-p}\,I, \quad K_1 = \sqrt{p}\,X

    where *p* \\in [0, 1] is the probability of a bit flip.
    """

    _num_wires = 1
    _param_names = ("p",)

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
            List ``[K0, K1]`` where K0 = \\sqrt (1-p)·I and K1 = \\sqrt p·X.
        """
        p = self.p
        K0 = jnp.sqrt(1 - p) * Id._matrix
        K1 = jnp.sqrt(p) * PauliX._matrix
        return [K0, K1]


class PhaseFlip(KrausChannel):
    r"""Single-qubit phase-flip (Pauli-Z) error channel.

    .. math::
        K_0 = \sqrt{1-p}\,I, \quad K_1 = \sqrt{p}\,Z

    where *p* \\in [0, 1] is the probability of a phase flip.
    """

    _num_wires = 1
    _param_names = ("p",)

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
            List ``[K0, K1]`` where K0 = \\sqrt (1-p)·I and K1 = \\sqrt p·Z.
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

    where *p* \\in [0, 1].  At p = 3/4 the channel is fully depolarizing.
    """

    _num_wires = 1
    _param_names = ("p",)

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

    where *\\gamma* \\in [0, 1] is the probability of
    energy loss (|1\\rangle -> |0\\rangle).
    """

    _num_wires = 1
    _param_names = ("gamma",)

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
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=_cdtype())
        K1 = jnp.array([[0.0, jnp.sqrt(g)], [0.0, 0.0]], dtype=_cdtype())
        return [K0, K1]


class PhaseDamping(KrausChannel):
    r"""Single-qubit phase damping (dephasing) channel.

    .. math::
        K_0 = \begin{pmatrix}1 & 0\\ 0 & \sqrt{1-\gamma}\end{pmatrix},\quad
        K_1 = \begin{pmatrix}0 & 0\\ 0 & \sqrt{\gamma}\end{pmatrix}

    where *\\gamma* \\in [0, 1] is the phase damping probability.
    """

    _num_wires = 1
    _param_names = ("gamma",)

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
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=_cdtype())
        K1 = jnp.array([[0.0, 0.0], [0.0, jnp.sqrt(g)]], dtype=_cdtype())
        return [K0, K1]


class ThermalRelaxationError(KrausChannel):
    r"""Single-qubit thermal relaxation error channel.

    Models simultaneous T_1 energy relaxation and T_2 dephasing.  Two regimes
    are handled:

    T_2 <= T_1 (Markovian dephasing + reset):
        Six Kraus operators built from p_z (phase-flip probability), p_r0
        (reset-to-|0\\rangle probability) and p_r1 (reset-to-|1\\rangle probability).

    T_2 > T_1 (non-Markovian; Choi matrix decomposition):
        The Choi matrix is assembled from the relaxation/dephasing rates, then
        diagonalised; Kraus operators are K_i = \sqrt \lambda_i · mat(v_i).

    Attributes:
        pe: Excited-state population (thermal population of |1\\rangle).
        t1: T_1 longitudinal relaxation time.
        t2: T_2 transverse dephasing time.
        tg: Gate duration.
    """

    _num_wires = 1
    _param_names = ("pe", "t1", "t2", "tg")

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
            pe: Excited-state population (thermal population of |1\\rangle), in [0, 1].
            t1: T_1 longitudinal relaxation time, must be > 0.
            t2: T_2 transverse dephasing time, must be > 0 and <= 2·T_1.
            tg: Gate duration, must be >= 0.
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
            raise ValueError("t2 must be <= 2·t1.")
        if tg < 0:
            raise ValueError("tg must be >= 0.")
        self.pe = pe
        self.t1 = t1
        self.t2 = t2
        self.tg = tg
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the Kraus operators for the thermal relaxation channel.

        The number of operators depends on the regime:

        * T_2 <= T_1: six operators (identity, phase-flip, two reset-to-|0\\rangle,
          two reset-to-|1\\rangle).
        * T_2 > T_1: four operators derived from the Choi matrix eigendecomposition.

        Returns:
            List of 2x2 JAX arrays representing the Kraus operators.
        """
        pe, t1, t2, tg = self.pe, self.t1, self.t2, self.tg

        eT1 = jnp.exp(-tg / t1)
        p_reset = 1.0 - eT1
        eT2 = jnp.exp(-tg / t2)

        if t2 <= t1:
            # --- Case T_2 <= T_1: six Kraus operators ---
            pz = (1.0 - p_reset) * (1.0 - eT2 / eT1) / 2.0
            pr0 = (1.0 - pe) * p_reset
            pr1 = pe * p_reset
            pid = 1.0 - pz - pr0 - pr1

            K0 = jnp.sqrt(pid) * jnp.eye(2, dtype=_cdtype())
            K1 = jnp.sqrt(pz) * jnp.array([[1, 0], [0, -1]], dtype=_cdtype())
            K2 = jnp.sqrt(pr0) * jnp.array([[1, 0], [0, 0]], dtype=_cdtype())
            K3 = jnp.sqrt(pr0) * jnp.array([[0, 1], [0, 0]], dtype=_cdtype())
            K4 = jnp.sqrt(pr1) * jnp.array([[0, 0], [1, 0]], dtype=_cdtype())
            K5 = jnp.sqrt(pr1) * jnp.array([[0, 0], [0, 1]], dtype=_cdtype())
            return [K0, K1, K2, K3, K4, K5]

        else:
            # --- Case T_2 > T_1: Choi matrix decomposition ---
            # Choi matrix (column-major / reshaping convention matching PennyLane)
            choi = jnp.array(
                [
                    [1 - pe * p_reset, 0, 0, eT2],
                    [0, pe * p_reset, 0, 0],
                    [0, 0, (1 - pe) * p_reset, 0],
                    [eT2, 0, 0, 1 - (1 - pe) * p_reset],
                ],
                dtype=_cdtype(),
            )
            eigenvalues, eigenvectors = jnp.linalg.eigh(choi)
            # Each eigenvector (column of eigenvectors) reshaped as 2x2 -> one Kraus op
            kraus = []
            for i in range(4):
                lam = eigenvalues[i]
                vec = eigenvectors[:, i]
                mat = jnp.sqrt(jnp.abs(lam)) * vec.reshape(2, 2, order="F")
                kraus.append(mat.astype(_cdtype()))
            return kraus


class QubitChannel(KrausChannel):
    """Generic Kraus channel from a user-supplied list of Kraus operators.

    This replaces PennyLane's ``qml.QubitChannel`` and accepts an arbitrary set
    of Kraus matrices satisfying \\sigma_k K_k\\dagger K_k = I.

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
                of dimension ``2**k x 2**k`` where k = ``len(wires)``.
            wires: Qubit index or list of qubit indices this channel acts on.
        """
        self._kraus_ops = [jnp.asarray(K, dtype=_cdtype()) for K in kraus_ops]
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the stored Kraus operators.

        Returns:
            List of Kraus operator matrices.
        """
        return self._kraus_ops


# Single-qubit Pauli matrices (plain arrays, no Operation overhead)
_PAULI_MATS = [Id._matrix, PauliX._matrix, PauliY._matrix, PauliZ._matrix]
_PAULI_LABELS = ["I", "X", "Y", "Z"]
_PAULI_CLASSES = [Id, PauliX, PauliY, PauliZ]


def evolve_pauli_with_clifford(
    clifford: Operation,
    pauli: Operation,
    adjoint_left: bool = True,
) -> Operation:
    """Compute C\\dagger P C  (or  C P C\\dagger)  and
    return the result as an Operation.

    Both operators are first embedded into the full Hilbert space spanned by
    the union of their wire sets.  The result is wrapped in a
    :class:`Hermitian` so it can be used in further algebra.

    Args:
        clifford: A Clifford gate.
        pauli: A Pauli / Hermitian operator.
        adjoint_left: If ``True``, compute C\\dagger P C; otherwise C P C\\dagger.

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
    # Full matrix = mat \\otimes I_{missing}
    full_mat = mat
    for _ in missing:
        full_mat = jnp.kron(full_mat, jnp.eye(2, dtype=_cdtype()))

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

    For an n-qubit matrix (``2**n x 2**n``), returns the dominant Pauli
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
    n_qubits = int(jnp.round(jnp.log2(dim)))

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
        # Multi-qubit tensor product -> Hermitian with pauli label attached
        P = _reduce(jnp.kron, [_PAULI_MATS[i] for i in best_label])
        result_op = Hermitian(matrix=P, wires=wire_order, record=False)
        result_op._pauli_label = pauli_label
        return best_coeff, result_op


def pauli_string_from_operation(op: Operation) -> str:
    """Extract a Pauli word string from an operation.

    Maps ``PauliX`` -> ``"X"``, ``PauliY`` -> ``"Y"``, ``PauliZ`` -> ``"Z"``,
    ``I`` -> ``"I"``.  For :class:`PauliRot`, returns its stored ``pauli_word``.
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
