"""Pulse/gate-independent entry point for building and simulating circuits.

This module is the main interaction point for manually creating circuits.  It
exposes the :class:`~qml_essentials.script.Script` circuit container, the
:func:`Hamiltonian` factory for time-evolution sources, and a few general
(pulse/gate-independent) quantum-info utilities.

Time evolution is invoked as a method on the Hamiltonian object::

    H = Hamiltonian(matrix, wires=0)          # static  -> Hermitian
    H_t = coeff_fn * Hamiltonian(matrix, 0)   # time-dep -> ParametrizedHamiltonian
    H_t.evolve(name="RX")([params], t)        # gate factory

The time-evolution engine itself lives in :mod:`qml_essentials.evolution` as
:class:`Evolution`, which is re-exported here for solver configuration
(``Evolution.set_solver_defaults`` / ``Evolution.clear_evolve_solver_cache``).
"""

from functools import reduce
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp

from qml_essentials.script import Script  # noqa: F401
from qml_essentials.evolution import Evolution  # noqa: F401
from qml_essentials.operations import (  # noqa: F401
    Hermitian,
    ParametrizedHamiltonian,
    PauliZ,
)


def Hamiltonian(
    matrix: jnp.ndarray,
    wires: Union[int, List[int]] = 0,
    record: bool = False,
) -> Hermitian:
    """Construct a (static) Hamiltonian as a :class:`Hermitian` operator.

    This is a thin factory over the existing :class:`Hermitian` operation —
    not a new type.  Multiply it by a coefficient function ``f(params, t)`` to
    obtain a time-dependent :class:`ParametrizedHamiltonian`.  Both expose an
    :meth:`evolve` method that returns a gate factory.

    Args:
        matrix: The Hermitian matrix defining this Hamiltonian.
        wires: Qubit index or list of qubit indices it acts on.
        record: Whether to record on the active tape.  Defaults to ``False``
            since a Hamiltonian used as an evolution source should not appear
            as a gate; the recorded operation is the one produced by
            :meth:`evolve`.

    Returns:
        A :class:`Hermitian` instance.
    """
    return Hermitian(matrix, wires=wires, record=record)


def _partial_trace_single(
    rho: jnp.ndarray,
    n_qubits: int,
    keep: List[int],
) -> jnp.ndarray:
    """Partial trace of a single density matrix (no batch dimension)."""
    shape = (2,) * (2 * n_qubits)
    rho_t = rho.reshape(shape)

    trace_out = sorted(set(range(n_qubits)) - set(keep))

    for q in reversed(trace_out):
        n_remaining = rho_t.ndim // 2
        rho_t = jnp.trace(rho_t, axis1=q, axis2=q + n_remaining)

    dim = 2 ** len(keep)
    return rho_t.reshape(dim, dim)


def partial_trace(
    rho: jnp.ndarray,
    n_qubits: int,
    keep: List[int],
) -> jnp.ndarray:
    """Partial trace of a density matrix, keeping only the specified qubits.

    Supports both single density matrices of shape ``(2**n, 2**n)`` and
    batched density matrices of shape ``(B, 2**n, 2**n)``.

    Args:
        rho: Density matrix of shape ``(2**n, 2**n)`` or ``(B, 2**n, 2**n)``.
        n_qubits: Total number of qubits.
        keep: List of qubit indices to *keep* (0-indexed).

    Returns:
        Reduced density matrix of shape ``(2**k, 2**k)`` or ``(B, 2**k, 2**k)``
        where *k* = ``len(keep)``.
    """

    dim = 2**n_qubits
    if rho.shape == (dim, dim):
        return _partial_trace_single(rho, n_qubits, keep)
    # Batched: shape (B, dim, dim)
    return jax.vmap(lambda r: _partial_trace_single(r, n_qubits, keep))(rho)


def _marginalize_probs_single(
    probs: jnp.ndarray,
    target_shape: Tuple[int],
    trace_out: Tuple[int],
) -> jnp.ndarray:
    """Marginalize a single probability vector (no batch dimension)."""
    probs_t = probs.reshape(target_shape)

    for q in trace_out:
        probs_t = probs_t.sum(axis=q)

    return probs_t.ravel()


def marginalize_probs(
    probs: jnp.ndarray,
    n_qubits: int,
    keep: Tuple[int],
) -> jnp.ndarray:
    """Marginalize a probability vector to keep only the specified qubits.

    Supports both single probability vectors of shape ``(2**n,)`` and
    batched vectors of shape ``(B, 2**n)``.

    Args:
        probs: Probability vector of shape ``(2**n,)`` or ``(B, 2**n)``.
        n_qubits: Total number of qubits.
        keep: List of qubit indices to *keep* (0-indexed).

    Returns:
        Marginalized probability vector of shape ``(2**k,)`` or ``(B, 2**k)``
        where *k* = ``len(keep)``.
    """

    dim = 2**n_qubits
    trace_out = tuple(q for q in range(n_qubits - 1, -1, -1) if q not in keep)
    target_shape = (2,) * n_qubits

    return jax.vmap(lambda p: _marginalize_probs_single(p, target_shape, trace_out))(
        probs.reshape(-1, dim)
    )


def build_parity_observable(
    qubit_group: List[int],
) -> Hermitian:
    """Build a multi-qubit parity observable.

    Args:
        qubit_group: List of qubit indices for the parity measurement.

    Returns:
        A :class:`Hermitian` operation whose matrix is the Z parity
        tensor product and whose wires match the given qubits.
    """
    Z = PauliZ._matrix
    mat = reduce(jnp.kron, [Z] * len(qubit_group))
    obs = Hermitian(matrix=mat, wires=qubit_group, record=False)
    # Tag the Pauli string so symbolic consumers (PauliWord / FourierTree) can
    # read it without an O(4^n) matrix decomposition.
    obs._pauli_label = "Z" * len(qubit_group)
    return obs
