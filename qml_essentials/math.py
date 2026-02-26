import jax
import jax.numpy as jnp
from qml_essentials.operations import _cdtype
from scipy.linalg import logm


def logm_v(A: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Compute the logarithm of a matrix. If the provided matrix has an additional
    batch dimension, the logarithm of each matrix is computed.

    Args:
        A (jnp.ndarray): The (potentially batched) matrices of which to compute
        the logarithm.

    Returns:
        jnp.ndarray: The log matrices
    """
    # TODO: check warnings
    if len(A.shape) == 2:
        return logm(A, **kwargs)
    elif len(A.shape) == 3:
        AV = jnp.zeros(A.shape, dtype=_cdtype())
        for i in range(A.shape[0]):
            AV = AV.at[i].set(logm(A[i], **kwargs))
        return AV
    else:
        raise NotImplementedError("Unsupported shape of input matrix")


def _sqrt_matrix(density_matrix: jnp.ndarray) -> jnp.ndarray:
    r"""Compute the matrix square root of a density matrix.

    Uses eigendecomposition: if :math:`\rho = V \Lambda V^\dagger`, then
    :math:`\sqrt{\rho} = V \sqrt{\Lambda} V^\dagger`.

    Negative eigenvalues (numerical noise) are clamped to zero.

    Args:
        density_matrix: Density matrix of shape ``(d, d)`` or ``(B, d, d)``.

    Returns:
        The matrix square root with the same shape as the input.
    """
    evs, vecs = jnp.linalg.eigh(density_matrix)
    evs = jnp.real(evs)
    evs = jnp.where(evs > 0.0, evs, 0.0)

    if density_matrix.ndim == 3:
        # batched: (B, d, d)
        sqrt_evs = jnp.sqrt(evs)[:, :, None] * jnp.eye(
            density_matrix.shape[-1], dtype=_cdtype()
        )
        return vecs @ sqrt_evs @ jnp.conj(jnp.transpose(vecs, (0, 2, 1)))

    # single: (d, d)
    return vecs @ jnp.diag(jnp.sqrt(evs)) @ jnp.conj(vecs.T)


def _fidelity_statevector(
    state0: jnp.ndarray,
    state1: jnp.ndarray,
) -> jnp.ndarray:
    r"""Fidelity between two pure states (state vectors).

    .. math::

        F(\ket{\psi}, \ket{\phi}) = \left|\braket{\psi | \phi}\right|^2
    """
    batched0 = state0.ndim > 1
    batched1 = state1.ndim > 1

    idx0 = "ab" if batched0 else "b"
    idx1 = "ab" if batched1 else "b"
    target = "a" if (batched0 or batched1) else ""

    overlap = jnp.einsum(f"{idx0},{idx1}->{target}", state0, jnp.conj(state1))
    return jnp.abs(overlap) ** 2


def _fidelity_dm(
    state0: jnp.ndarray,
    state1: jnp.ndarray,
) -> jnp.ndarray:
    r"""Fidelity between two mixed states (density matrices)."""
    sqrt_state0 = _sqrt_matrix(state0)
    product = sqrt_state0 @ state1 @ sqrt_state0

    evs = jnp.linalg.eigvalsh(product)
    evs = jnp.real(evs)
    evs = jnp.where(evs > 0.0, evs, 0.0)

    return jnp.sum(jnp.sqrt(evs), axis=-1) ** 2


def fidelity(
    state0: jnp.ndarray,
    state1: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the fidelity between two quantum states.

    Accepts either state vectors or density matrices.

    Args:
        state0: State vector or density matrix.
        state1: State vector or density matrix (same kind as *state0*).

    Returns:
        Fidelity (scalar or shape ``(B,)``).

    Raises:
        ValueError: If the two states have incompatible shapes or
            different representations (vector vs. matrix).
    """
    state0 = jnp.asarray(state0, dtype=_cdtype())
    state1 = jnp.asarray(state1, dtype=_cdtype())

    if state0.shape[-1] != state1.shape[-1]:
        raise ValueError("The two states must have the same number of wires.")

    is_sv0 = state0.ndim <= 2 and (
        state0.ndim == 1 or state0.shape[-2] != state0.shape[-1]
    )
    is_sv1 = state1.ndim <= 2 and (
        state1.ndim == 1 or state1.shape[-2] != state1.shape[-1]
    )

    if is_sv0 != is_sv1:
        raise ValueError(
            "Both states must be of the same kind "
            "(both state vectors or both density matrices)."
        )

    if is_sv0:
        return _fidelity_statevector(state0, state1)
    return _fidelity_dm(state0, state1)


def trace_distance(
    state0: jnp.ndarray,
    state1: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the trace distance between two quantum states.

    Supports single density matrices of shape ``(2**N, 2**N)`` and batched
    density matrices of shape ``(B, 2**N, 2**N)``.

    Args:
        state0: Density matrix of shape ``(2**N, 2**N)`` or ``(B, 2**N, 2**N)``.
        state1: Density matrix of shape ``(2**N, 2**N)`` or ``(B, 2**N, 2**N)``.

    Returns:
        Trace distance (scalar or shape ``(B,)``).
    """
    state0 = jnp.asarray(state0, dtype=_cdtype())
    state1 = jnp.asarray(state1, dtype=_cdtype())

    if state0.shape[-1] != state1.shape[-1]:
        raise ValueError("The two states must have the same number of wires.")

    eigvals = jnp.abs(jnp.linalg.eigvalsh(state0 - state1))
    return jnp.sum(eigvals, axis=-1) / 2


def phase_difference(
    state0: jnp.ndarray,
    state1: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the phase difference between two state vectors.

    A value of zero indicates the two states are related by at most a
    real global factor (i.e. no relative phase).  The result lies in
    :math:`[0, 1 + \pi]`.

    Supports single state vectors of shape ``(2**N,)`` and batched state
    vectors of shape ``(B, 2**N)``.

    Args:
        state0: State vector of shape ``(2**N,)`` or ``(B, 2**N)``.
        state1: State vector of shape ``(2**N,)`` or ``(B, 2**N)``.

    Returns:
        Phase difference (scalar or shape ``(B,)``).
    """
    state0 = jnp.asarray(state0, dtype=_cdtype())
    state1 = jnp.asarray(state1, dtype=_cdtype())

    if state0.shape[-1] != state1.shape[-1]:
        raise ValueError("The two states must have the same number of wires.")

    batched0 = state0.ndim > 1
    batched1 = state1.ndim > 1

    idx0 = "ab" if batched0 else "b"
    idx1 = "ab" if batched1 else "b"
    target = "a" if (batched0 or batched1) else ""

    inner = jnp.einsum(f"{idx0},{idx1}->{target}", jnp.conj(state0), state1)
    return jnp.abs(1.0 - jnp.angle(inner))
