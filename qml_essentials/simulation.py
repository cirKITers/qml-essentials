"""Pure simulation and measurement kernels for :class:`~qml_essentials.script.Script`.

These functions are stateless: they take a recorded tape (a list of
:class:`~qml_essentials.operations.Operation`) plus measurement parameters and
return JAX arrays.  Keeping them as module-level free functions (rather than
static methods on ``Script``) makes the simulation engine independently testable
and keeps ``script.py`` focused on orchestration.
"""

from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np  # needed to prevent jitting some operations

from qml_essentials.operations import (
    Barrier,
    Operation,
    KrausChannel,
    _einsum_subscript,
    _cdtype,
)


def infer_n_qubits(ops: List[Operation], obs: List[Operation]) -> int:
    """Infer the number of qubits from a list of operations and observables.

    Args:
        ops: Gate operations recorded on the tape.
        obs: Observable operations used for measurement.

    Returns:
        The smallest number of qubits that covers all wire indices, i.e.
        ``max(all_wires) + 1`` (at least 1).
    """
    all_wires: set[int] = set()
    for op in ops + obs:
        all_wires.update(op.wires)
    return max(all_wires) + 1 if all_wires else 1


def uses_density(tape: List[Operation], type: str) -> bool:
    """Return whether density-matrix simulation is required.

    Density-matrix simulation is needed when the caller explicitly requests the
    ``"density"`` measurement type, or when the tape contains a noise channel
    (a :class:`~qml_essentials.operations.KrausChannel`).

    Args:
        tape: Ordered list of gate/channel operations.
        type: Requested measurement type.

    Returns:
        ``True`` if density-matrix simulation must be used.
    """
    has_noise = any(isinstance(op, KrausChannel) for op in tape)
    return type == "density" or has_noise


def _stack_obs(obs: List[Operation], n_qubits: int) -> jnp.ndarray:
    """Stack lifted observable matrices into a single ``(n_obs, dim, dim)`` array."""
    return jnp.stack([ob.lifted_matrix(n_qubits) for ob in obs], axis=0)


def simulate_pure(
    tape: List[Operation],
    n_qubits: int,
    initial_state: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Statevector simulation kernel.

    Starts from |00…0⟩ (or *initial_state* when given) and applies each gate in
    *tape* via tensor contraction.  The state is kept in rank-*n* tensor form
    ``(2,)*n`` throughout the gate loop to avoid per-gate ``reshape`` dispatch;
    only the initial and final conversions to/from the flat ``(2**n,)``
    representation incur a reshape.

    All gate tensors and einsum subscript strings are pre-extracted from
    the tape before the simulation loop so that each iteration performs
    only a single ``jnp.einsum`` call with zero additional Python
    overhead (no method dispatch, no property access, no cache lookup).

    Args:
        tape: Ordered list of gate operations to apply.
        n_qubits: Total number of qubits.
        initial_state: Optional statevector of shape ``(2**n_qubits,)`` to start
            from.  When ``None`` (default), the all-zero state |00…0⟩ is used.

    Returns:
        Statevector of shape ``(2**n_qubits,)``.
    """
    dim = 2**n_qubits

    # Pre-extract gate tensors and einsum subscripts — eliminates all
    # per-gate Python overhead (method calls, property lookups, cache
    # hits on _einsum_subscript) from the hot loop.
    compiled = []
    for op in tape:
        if isinstance(op, Barrier):
            continue
        k = len(op.wires)
        gt = op._gate_tensor(k)
        sub = _einsum_subscript(n_qubits, k, tuple(op.wires))
        compiled.append((gt, sub))

    if initial_state is None:
        state = jnp.zeros(dim, dtype=_cdtype()).at[0].set(1.0)
    else:
        state = jnp.asarray(initial_state, dtype=_cdtype()).reshape(dim)
    psi = state.reshape((2,) * n_qubits)
    for gt, sub in compiled:
        psi = jnp.einsum(sub, gt, psi)
    return psi.reshape(dim)


def simulate_mixed(
    tape: List[Operation],
    n_qubits: int,
    initial_state: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Density-matrix simulation kernel.

    Starts from \\rho  = \\vert 0\\rangle\\langle 0\\vert (or from
    \\rho  = \\vert\\psi\\rangle\\langle\\psi\\vert for a given *initial_state*
    \\vert\\psi\\rangle) and applies each gate in *tape* via
    :meth:`~qml_essentials.operations.Operation.apply_to_density`
    (\\rho  -> U\\rho U† for unitaries, \\Sigma_k K_k \\rho  K_k\\dagger
    for Kraus channels).
    Required for noisy circuits.

    Args:
        tape: Ordered list of gate or channel operations to apply.
        n_qubits: Total number of qubits.
        initial_state: Optional statevector of shape ``(2**n_qubits,)`` to start
            from.  When ``None`` (default), the all-zero state |00…0⟩ is used.

    Returns:
        Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
    """
    dim = 2**n_qubits
    if initial_state is None:
        rho = jnp.zeros((dim, dim), dtype=_cdtype()).at[0, 0].set(1.0)
    else:
        psi = jnp.asarray(initial_state, dtype=_cdtype()).reshape(dim)
        rho = jnp.outer(psi, jnp.conj(psi))
    for op in tape:
        rho = op.apply_to_density(rho, n_qubits)
    return rho


def simulate_and_measure(
    tape: List[Operation],
    n_qubits: int,
    type: str,
    obs: List[Operation],
    use_density: bool,
    shots: Optional[int] = None,
    key: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Run simulation and measurement in a single dispatch.

    Chooses statevector or density-matrix simulation based on
    *use_density*, then applies the appropriate measurement function.
    This eliminates duplicated branching logic in single-sample and
    batched execution paths.

    When *shots* is not ``None``, the exact probability distribution is
    first computed, then ``shots`` samples are drawn from it to produce
    a noisy estimate of the requested measurement (``"probs"`` or
    ``"expval"``).

    Pure-circuit density optimisation — when ``type == "density"``
    but no noise channels are present on the tape, the density matrix
    is computed via statevector simulation followed by an outer product
    ``\\rho  = \\vert\\psi\\rangle\\langle\\psi\\vert``
    instead of evolving the full ``2^n\\times 2^n`` matrix
    gate by gate.  This reduces the per-gate cost from O(4^n) to
    O(2^n), giving a significant speed-up for medium qubit counts
    (~4x for 5 qubits).

    Args:
        tape: Ordered list of gate/channel operations to apply.
        n_qubits: Total number of qubits.
        type: Measurement type (``"state"``/``"probs"``/``"expval"``/
            ``"density"``).
        obs: Observables for ``"expval"`` measurements.
        use_density: If ``True``, use density-matrix simulation.
        shots: Number of measurement shots.  If ``None`` (default),
            exact analytic results are returned.
        key: JAX PRNG key for shot sampling.  Required when *shots*
            is not ``None``.
        initial_state: Optional statevector of shape ``(2**n_qubits,)`` to start
            from.  When ``None`` (default), the all-zero state |00…0⟩ is used.

    Returns:
        Measurement result (shape depends on *type*).
    """
    if use_density:
        # Check if any operation is actually a noise channel.
        has_noise = any(isinstance(o, KrausChannel) for o in tape)
        if has_noise:
            # Must do full density-matrix evolution for Kraus channels.
            rho = simulate_mixed(tape, n_qubits, initial_state=initial_state)
        else:
            # Pure circuit requesting density output: simulate the
            # statevector (O(depth\times 2^n)) and form  # noqa: W605
            # \rho  = \vert\psi\rangle\langle\psi\vert once  # noqa: W605
            # (O(4^n)).  This avoids the O(depth\times 4^n) cost of  # noqa: W605
            # evolving the full density matrix gate by gate.
            state = simulate_pure(tape, n_qubits, initial_state=initial_state)
            rho = jnp.outer(state, jnp.conj(state))

        if shots is not None and type in ("probs", "expval"):
            exact_probs = jnp.real(jnp.diag(rho))
            return sample_shots(exact_probs, n_qubits, type, obs, shots, key)
        return measure_density(rho, n_qubits, type, obs)

    state = simulate_pure(tape, n_qubits, initial_state=initial_state)

    if shots is not None and type in ("probs", "expval"):
        exact_probs = jnp.abs(state) ** 2
        return sample_shots(exact_probs, n_qubits, type, obs, shots, key)
    return measure_state(state, n_qubits, type, obs)


def measure_state(
    state: jnp.ndarray,
    n_qubits: int,
    type: str,
    obs: List[Operation],
) -> jnp.ndarray:
    """Apply the requested measurement to a pure statevector.

    Args:
        state: Statevector of shape ``(2**n_qubits,)``.
        n_qubits: Total number of qubits.
        type: Measurement type — one of ``"state"``, ``"probs"``,
            or ``"expval"``.
        obs: Observables used when *type* is ``"expval"``.

    Returns:
        Measurement result whose shape depends on *type*:

        - ``"state"``  -> ``(2**n_qubits,)``
        - ``"probs"``  -> ``(2**n_qubits,)``
        - ``"expval"`` -> ``(len(obs),)``

    Raises:
        ValueError: If *type* is not a recognised measurement type.
    """
    if type == "state":
        return state

    if type == "probs":
        return jnp.abs(state) ** 2

    if type == "expval":
        # Fast path for single-qubit diagonal observables (PauliZ, etc.)
        # where d0, d1 are the diagonal elements of the 2x2 observable.
        # This replaces n_obs tensor contractions with a single |ψ|²
        # and n_obs reductions over the probability vector.

        def _is_single_qubit_diag(ob):
            m = ob.__class__._matrix
            if m is None or len(ob.wires) != 1:
                return False
            # Convert to NumPy to ensure concrete boolean evaluation
            m_np = np.asarray(m)
            return np.allclose(m_np - np.diag(np.diag(m_np)), 0)

        all_single_qubit_diag = all(_is_single_qubit_diag(ob) for ob in obs)

        if all_single_qubit_diag:
            probs = jnp.abs(state) ** 2
            psi_t = probs.reshape((2,) * n_qubits)
            results = []
            for ob in obs:
                q = ob.wires[0]
                d = np.real(np.diag(np.asarray(ob.__class__._matrix)))
                # Sum probabilities over all axes except qubit q
                p_q = jnp.sum(psi_t, axis=tuple(i for i in range(n_qubits) if i != q))
                results.append(d[0] * p_q[0] + d[1] * p_q[1])
            return jnp.array(results)

        # General path: stack observable matrices and use a single
        # batched matmul instead of a Python loop of tensor contractions.
        # O_states[i] = obs[i] |ψ⟩, then ⟨O_i⟩ = Re(⟨ψ|O_states[i]⟩).
        obs_mats = _stack_obs(obs, n_qubits)  # (n_obs, dim, dim)
        # Batched matvec: (n_obs, dim, dim) @ (dim,) -> (n_obs, dim)
        O_states = jnp.einsum("oij,j->oi", obs_mats, state)
        return jnp.real(jnp.einsum("i,oi->o", jnp.conj(state), O_states))

    raise ValueError(f"Unknown measurement type: {type!r}")


def measure_density(
    rho: jnp.ndarray,
    n_qubits: int,
    type: str,
    obs: List[Operation],
) -> jnp.ndarray:
    """Apply the requested measurement to a density matrix.

    Args:
        rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        n_qubits: Total number of qubits.
        type: Measurement type — one of ``"density"``, ``"probs"``,
            or ``"expval"``.
        obs: Observables used when *type* is ``"expval"``.

    Returns:
        Measurement result whose shape depends on *type*:

        - ``"density"`` -> ``(2**n_qubits, 2**n_qubits)``
        - ``"probs"``   -> ``(2**n_qubits,)``
        - ``"expval"``  -> ``(len(obs),)``

    Raises:
        ValueError: If *type* is ``"state"`` (not valid for mixed circuits)
            or another unrecognised type.
    """
    if type == "density":
        return rho

    if type == "probs":
        return jnp.real(jnp.diag(rho))

    if type == "expval":
        # Tr(O \\rho ) = \\Sigma_ij O_ij \\rho _ji
        # Stack all observable matrices and compute all traces in one
        # batched operation.
        obs_mats = _stack_obs(obs, n_qubits)  # (n_obs, dim, dim)
        # einsum "oij,ji->o" computes Tr(O_o @ \\rho ) for each observable
        return jnp.real(jnp.einsum("oij,ji->o", obs_mats, rho))

    raise ValueError(
        "Measurement type 'state' is not defined for mixed (noisy) circuits. "
        "Use 'density' instead."
    )


def sample_shots(
    probs: jnp.ndarray,
    n_qubits: int,
    type: str,
    obs: List[Operation],
    shots: int,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Convert exact probabilities into shot-sampled results.

    Draws *shots* samples from the computational-basis probability
    distribution and returns either estimated probabilities or
    shot-based expectation values.

    Args:
        probs: Exact probability vector of shape ``(2**n_qubits,)``.
        n_qubits: Total number of qubits.
        type: Measurement type — ``"probs"`` or ``"expval"``.
        obs: Observables used when *type* is ``"expval"``.
        shots: Number of measurement shots.
        key: JAX PRNG key for sampling.

    Returns:
        Shot-sampled measurement result:

        - ``"probs"``  → ``(2**n_qubits,)`` estimated probabilities.
        - ``"expval"`` → ``(len(obs),)`` estimated expectation values.
    """
    dim = 2**n_qubits

    # Draw `shots` samples from the computational basis.
    # Each sample is an integer in [0, dim) representing a basis state.
    samples = jax.random.choice(key, dim, shape=(shots,), p=probs)

    # Build a histogram of counts for each basis state.
    counts = jnp.zeros(dim, dtype=jnp.int32)
    counts = counts.at[samples].add(1)
    estimated_probs = counts / shots

    if type == "probs":
        return estimated_probs

    if type == "expval":
        # For each observable, compute O from the shot-sampled
        # probabilities.  For diagonal observables this is exact;
        # for general observables we use Tr(O · diag(estimated_probs)).
        results = []
        for ob in obs:
            O_mat = ob.lifted_matrix(n_qubits)
            # diagonal approximation from
            # computational basis measurements, which is exact for
            # diagonal observables like PauliZ)
            results.append(jnp.real(jnp.dot(jnp.diag(O_mat), estimated_probs)))
        return jnp.array(results)

    raise ValueError(
        f"Shot simulation is only supported for 'probs' and 'expval', got {type!r}."
    )
