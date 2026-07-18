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
    The inputs are normalised before the overlap is computed so that
    the result is always in :math:`[0, 1]`.
    """
    # Normalise so that unnormalised inputs don't produce F > 1.
    norm0 = jnp.linalg.norm(state0, axis=-1, keepdims=True)
    norm1 = jnp.linalg.norm(state1, axis=-1, keepdims=True)
    state0 = state0 / jnp.where(norm0 > 0, norm0, 1.0)
    state1 = state1 / jnp.where(norm1 > 0, norm1, 1.0)

    batched0 = state0.ndim > 1
    batched1 = state1.ndim > 1

    idx0 = "ab" if batched0 else "b"
    idx1 = "ab" if batched1 else "b"
    target = "a" if (batched0 or batched1) else ""

    overlap = jnp.einsum(f"{idx0},{idx1}->{target}", jnp.conj(state0), state1)
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
    :math:`[-\pi, 1 + \pi]`.

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
    return jnp.angle(inner)


def _fubini_study_statevector(
    jac: jnp.ndarray,
    state: jnp.ndarray,
) -> jnp.ndarray:
    r"""Fubini-Study metric of a pure state.

    The Fubini-Study metric is the real part of the quantum geometric tensor:

    .. math::

        g_{ij} = \mathrm{Re}\left[
            \braket{\partial_i\psi | \partial_j\psi}
            - \braket{\partial_i\psi | \psi}\braket{\psi | \partial_j\psi}
        \right]

    It relates to the pure-state quantum Fisher information by
    :math:`F_{ij} = 4\,g_{ij}`.

    Args:
        jac: Jacobian :math:`\partial\ket{\psi}/\partial\theta` of shape
            ``(d, P)`` with ``d = 2**N`` and ``P`` the number of parameters.
        state: State vector of shape ``(d,)``.

    Returns:
        Real, symmetric metric of shape ``(P, P)``.
    """
    A = jnp.conj(jac.T) @ jac  # A_ij = <∂_i ψ | ∂_j ψ>
    v = jnp.conj(jac.T) @ state  # v_i = <∂_i ψ | ψ>
    return jnp.real(A - jnp.outer(v, jnp.conj(v)))


def _qfi_statevector(
    jac: jnp.ndarray,
    state: jnp.ndarray,
) -> jnp.ndarray:
    r"""Quantum Fisher Information of a pure state.

    For a normalised state :math:`\ket{\psi(\theta)}` the QFI is four times the
    Fubini-Study metric (see :func:`_fubini_study_statevector`):

    .. math::

        F_{ij} = 4\,\mathrm{Re}\left[
            \braket{\partial_i\psi | \partial_j\psi}
            - \braket{\partial_i\psi | \psi}\braket{\psi | \partial_j\psi}
        \right]

    Args:
        jac: Jacobian :math:`\partial\ket{\psi}/\partial\theta` of shape
            ``(d, P)`` with ``d = 2**N`` and ``P`` the number of parameters.
        state: State vector of shape ``(d,)``.

    Returns:
        Real, symmetric QFI matrix of shape ``(P, P)``.
    """
    return 4.0 * _fubini_study_statevector(jac, state)


def _qfi_density(
    jac: jnp.ndarray,
    state: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    r"""Quantum Fisher Information of a mixed state.

    Using the symmetric logarithmic derivative, the QFI of a density matrix
    :math:`\rho = \sum_k p_k \ket{k}\bra{k}` reads

    .. math::

        F_{ij} = 2 \sum_{k, l : p_k + p_l > 0}
            \frac{\mathrm{Re}\left(
                \braket{k | \partial_i\rho | l}\braket{l | \partial_j\rho | k}
            \right)}{p_k + p_l}

    Eigenvalue pairs with :math:`p_k + p_l \le` ``eps`` are excluded from the
    sum. Negative eigenvalues (numerical noise) are clamped to zero.

    Args:
        jac: Jacobian :math:`\partial\rho/\partial\theta` of shape
            ``(d, d, P)`` with ``d = 2**N`` and ``P`` the number of parameters.
        state: Density matrix of shape ``(d, d)``.
        eps: Threshold below which an eigenvalue pair is masked out.

    Returns:
        Real, symmetric QFI matrix of shape ``(P, P)``.
    """
    evals, evecs = jnp.linalg.eigh(state)
    evals = jnp.where(jnp.real(evals) > 0.0, jnp.real(evals), 0.0)

    # ∂_i ρ in the eigenbasis: M[i]_kl = <k| ∂_i ρ |l>.
    drho = jnp.moveaxis(jac, -1, 0)  # (P, d, d)
    M = jnp.conj(evecs.T) @ drho @ evecs  # broadcast (d, d) over P

    s = evals[:, None] + evals[None, :]  # (d, d)
    weights = jnp.where(s > eps, 2.0 / s, 0.0)

    # ∂_i ρ is Hermitian, so <l| ∂_j ρ |k> = conj(<k| ∂_j ρ |l>).
    F = jnp.einsum("ikl,jkl->ij", M * weights[None], jnp.conj(M))
    return jnp.real(F)


def _state_and_jacobian(state_fn, params: jnp.ndarray):
    r"""Evaluate *state_fn* and its Jacobian at *params*.

    The Jacobian is obtained with forward-mode automatic differentiation
    (:func:`jax.jacfwd`), which yields the complex Jacobian directly for the
    real-valued parameters.

    Args:
        state_fn: Callable mapping *params* to a quantum state.
        params: Parameters at which to evaluate.

    Returns:
        Tuple ``(state, jac)`` of the state and its Jacobian, both cast to the
        complex working dtype.
    """
    state = jnp.asarray(state_fn(params), dtype=_cdtype())
    jac = jnp.asarray(jax.jacfwd(state_fn)(params), dtype=_cdtype())
    return state, jac


def quantum_fisher_information(
    state_fn,
    params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the Quantum Fisher Information (QFI) at a parameter point.

    The QFI is the metric tensor of the state manifold evaluated at
    ``params``. It therefore requires the state as a *function* of the
    parameters rather than a single state; the Jacobian is obtained with
    forward-mode automatic differentiation (:func:`jax.jacfwd`), which yields
    the complex Jacobian directly for real-valued parameters.

    Both pure and mixed states are supported and dispatched on the kind of
    object returned by *state_fn* (state vector vs. density matrix), mirroring
    :func:`fidelity`:

    - state vector of shape ``(d,)`` -> Fubini-Study formula
      (see :func:`_qfi_statevector`),
    - density matrix of shape ``(d, d)`` -> symmetric logarithmic derivative
      formula (see :func:`_qfi_density`).

    The returned matrix has shape ``(P, P)`` where ``P`` is the total number of
    parameters (the parameter axes of *params* are flattened).

    Args:
        state_fn: Callable mapping *params* to a normalised quantum state.
            Typically ``lambda p: model(params=p, inputs=x)`` with the model's
            ``execution_type`` set to ``"state"`` (pure) or ``"density"``
            (mixed).
        params: Parameters at which the QFI is evaluated. Must be passed in the
            shape expected by *state_fn* (e.g. the model's batched
            ``model.params``).

    Returns:
        Real, symmetric QFI matrix of shape ``(P, P)``.

    Raises:
        ValueError: If *state_fn* returns neither a state vector nor a square
            density matrix.
    """
    state, jac = _state_and_jacobian(state_fn, params)

    if state.ndim == 1:
        jac = jac.reshape(state.shape[0], -1)
        return _qfi_statevector(jac, state)
    elif state.ndim == 2 and state.shape[-1] == state.shape[-2]:
        jac = jac.reshape(state.shape[0], state.shape[1], -1)
        return _qfi_density(jac, state)
    else:
        raise ValueError(
            "state_fn must return a state vector of shape (d,) or a density "
            f"matrix of shape (d, d), got shape {state.shape}."
        )


def fubini_study_metric(
    state_fn,
    params: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the Fubini-Study metric tensor at a parameter point.

    The Fubini-Study metric is the real part of the quantum geometric tensor on
    the manifold of pure states and equals the pure-state quantum Fisher
    information up to a factor of four, :math:`F_{ij} = 4\,g_{ij}`:

    .. math::

        g_{ij} = \mathrm{Re}\left[
            \braket{\partial_i\psi | \partial_j\psi}
            - \braket{\partial_i\psi | \psi}\braket{\psi | \partial_j\psi}
        \right]

    It is only defined for pure states; *state_fn* must therefore return a
    normalised state vector. See :func:`quantum_fisher_information` for the
    calling convention.

    Args:
        state_fn: Callable mapping *params* to a normalised state vector.
            Typically ``lambda p: model(params=p, inputs=x)`` with the model's
            ``execution_type`` set to ``"state"``.
        params: Parameters at which the metric is evaluated.

    Returns:
        Real, symmetric metric of shape ``(P, P)`` where ``P`` is the total
        number of parameters.

    Raises:
        ValueError: If *state_fn* does not return a state vector.
    """
    state, jac = _state_and_jacobian(state_fn, params)

    if state.ndim != 1:
        raise ValueError(
            "The Fubini-Study metric is only defined for pure states; "
            f"state_fn must return a state vector of shape (d,), got shape "
            f"{state.shape}."
        )

    jac = jac.reshape(state.shape[0], -1)
    return _fubini_study_statevector(jac, state)
