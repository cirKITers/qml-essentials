"""Pulse/gate-independent entry point for building and simulating circuits.

This module is the main interaction point for manually creating circuits.  It
re-exports the :class:`~qml_essentials.script.Script` circuit container and a few
general (pulse/gate-independent) quantum-info utilities.

The Hamiltonian time-evolution machinery used to live here on a ``Jaqsi`` class;
it now resides in :mod:`qml_essentials.evolution` as :class:`Evolution`.  For
backward compatibility, :func:`evolve` is re-exported and ``Jaqsi`` is kept as an
alias for :class:`~qml_essentials.evolution.Evolution`.
"""

from functools import reduce
from typing import List, Tuple

import jax
import jax.numpy as jnp

from qml_essentials.script import Script  # noqa: F401
from qml_essentials.evolution import Evolution, evolve  # noqa: F401
from qml_essentials.operations import Hermitian, PauliZ

# Backward-compatible alias: the evolution/solver machinery moved to
# :mod:`qml_essentials.evolution`.  ``Jaqsi`` is retained as an alias for
# :class:`Evolution` so existing ``Jaqsi.<member>`` references — including the
# shared ``_evolve_solver_cache`` / ``_solver_defaults`` state — keep working.
Jaqsi = Evolution


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
    return Hermitian(matrix=mat, wires=qubit_group, record=False)
