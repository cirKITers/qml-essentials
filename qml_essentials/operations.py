from typing import List, Optional, Union

import jax
import jax.numpy as jnp

from qml_essentials.tape import active_tape, recording  # noqa: F401 (re-export)

# Enable 64-bit precision (important for quantum simulation accuracy)
jax.config.update("jax_enable_x64", True)


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

    def __init__(
        self,
        wires: Union[int, List[int]] = 0,
        matrix: Optional[jnp.ndarray] = None,
    ) -> None:
        """Initialise the operation and optionally register it on the active tape.

        Args:
            wires: Qubit index or list of qubit indices this operation acts on.
            matrix: Optional explicit gate matrix.  When provided it overrides
                the class-level ``_matrix`` attribute.
        """
        self.wires = wires if isinstance(wires, list) else [wires]
        if matrix is not None:
            self._matrix = matrix

        # If a tape is currently recording, append ourselves
        tape = active_tape()
        if tape is not None:
            tape.append(self)

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
        if isinstance(wires, int):
            self._wires = [wires]
        else:
            self._wires = wires

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
        """Apply this gate to a density matrix via ρ → UρU†.

        The density matrix (shape ``(2**n, 2**n)``) is treated as a rank-*2n*
        tensor with *n* "ket" axes (0..n-1) and *n* "bra" axes (n..2n-1).
        U acts on the ket half; U* acts on the bra half.  Both contractions
        reuse the same axis-permutation logic as :meth:`apply_to_state`,
        keeping the operation allocation-free with respect to building full
        unitaries.

        This is the correct building block for noise channels, where a mixed
        state cannot be represented as a pure statevector.

        Args:
            rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        k = len(self.wires)
        U = self.matrix.reshape((2,) * 2 * k)  # (out)*k + (in)*k
        U_conj = jnp.conj(U)

        # Represent ρ as a (2,)*2n tensor: axes 0..n-1 = ket, n..2n-1 = bra
        rho_t = rho.reshape((2,) * 2 * n_qubits)

        # Helper: apply a (2,)*2k gate tensor to k chosen axes of rho_t,
        # then restore the correct axis order for a 2n-rank tensor.
        def _contract_and_restore(
            tensor: jnp.ndarray,
            gate: jnp.ndarray,
            target_axes: List[int],
        ) -> jnp.ndarray:
            """Contract *gate* against *target_axes* of *tensor* and restore axis order.

            Args:
                tensor: Rank-``2*n_qubits`` tensor representing the density matrix.
                gate: Reshaped gate tensor of shape ``(2,)*2k``.
                target_axes: The *k* axes of *tensor* to contract against.

            Returns:
                Updated tensor with the same rank as *tensor*, with the
                contracted axes restored to their original positions.
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
    """A generic Hermitian observable or gate defined by an arbitrary matrix.

    Example:
        >>> obs = Hermitian(matrix=my_matrix, wires=0)
    """

    def __init__(self, matrix: jnp.ndarray, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Hermitian operator.

        Args:
            matrix: The Hermitian matrix defining this operator.
            wires: Qubit index or list of qubit indices this operator acts on.
        """
        super().__init__(wires=wires, matrix=jnp.asarray(matrix, dtype=jnp.complex128))


class I(Operation):
    """Identity gate."""

    _matrix = jnp.eye(2, dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an identity gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class PauliX(Operation):
    """Pauli-X gate / observable (bit-flip, σ_x)."""

    _matrix = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Pauli-X gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class PauliY(Operation):
    """Pauli-Y gate / observable (σ_y)."""

    _matrix = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Pauli-Y gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class PauliZ(Operation):
    """Pauli-Z gate / observable (phase-flip, σ_z)."""

    _matrix = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Pauli-Z gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class H(Operation):
    """Hadamard gate."""

    _matrix = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Hadamard gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class RX(Operation):
    """Rotation around the X axis: RX(θ) = exp(-i θ/2 X)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an RX rotation gate.

        Args:
            theta: Rotation angle in radians.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.theta = theta
        mat = jnp.cos(theta / 2) * I._matrix - 1j * jnp.sin(theta / 2) * PauliX._matrix
        super().__init__(wires=wires, matrix=mat)


class RY(Operation):
    """Rotation around the Y axis: RY(θ) = exp(-i θ/2 Y)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an RY rotation gate.

        Args:
            theta: Rotation angle in radians.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.theta = theta
        mat = jnp.cos(theta / 2) * I._matrix - 1j * jnp.sin(theta / 2) * PauliY._matrix
        super().__init__(wires=wires, matrix=mat)


class RZ(Operation):
    """Rotation around the Z axis: RZ(θ) = exp(-i θ/2 Z)."""

    def __init__(self, theta: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an RZ rotation gate.

        Args:
            theta: Rotation angle in radians.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.theta = theta
        mat = jnp.cos(theta / 2) * I._matrix - 1j * jnp.sin(theta / 2) * PauliZ._matrix
        super().__init__(wires=wires, matrix=mat)


class CX(Operation):
    """Controlled-X (CNOT) gate.

    Args on construction:
        wires: ``[control, target]``.
    """

    _matrix = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=jnp.complex128,
    )

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

    def __init__(self, wires: List[int] = [0, 1, 2]) -> None:
        """Initialise a Toffoli (CCX) gate.

        Args:
            wires: Three-element list ``[control0, control1, target]``.
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
        rz_phi = jnp.cos(phi / 2) * I._matrix - 1j * jnp.sin(phi / 2) * PauliZ._matrix
        ry_theta = (
            jnp.cos(theta / 2) * I._matrix - 1j * jnp.sin(theta / 2) * PauliY._matrix
        )
        rz_omega = (
            jnp.cos(omega / 2) * I._matrix - 1j * jnp.sin(omega / 2) * PauliZ._matrix
        )
        mat = rz_omega @ ry_theta @ rz_phi
        super().__init__(wires=wires, matrix=mat)


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

        Uses the same axis-permutation engine as
        :meth:`Operation.apply_to_density`, but sums over all Kraus operators.

        Args:
            rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
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
        K0 = jnp.sqrt(1 - p) * I._matrix
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
        K0 = jnp.sqrt(1 - p) * I._matrix
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
        K0 = jnp.sqrt(1 - p) * I._matrix
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
            List of 2×2 JAX arrays representing the Kraus operators.
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
            # Each eigenvector (column of eigenvectors) reshaped as 2×2 → one Kraus op
            kraus = []
            for i in range(4):
                lam = eigenvalues[i]
                vec = eigenvectors[:, i]
                # Map ℂ⁴ → ℂ²ˣ² with column-major order (matching PL convention)
                mat = jnp.sqrt(jnp.abs(lam)) * vec.reshape(2, 2, order="F")  # type: ignore[call-overload]
                kraus.append(mat.astype(jnp.complex128))
            return kraus
