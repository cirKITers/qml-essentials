from functools import reduce
from typing import Any, Callable, List, Optional, Tuple, Union

import diffrax
import jax
import jax.numpy as jnp
import jax.scipy.linalg

from qml_essentials.operations import (
    Hermitian,
    ParametrizedHamiltonian,
    Operation,
    KrausChannel,
    PauliZ,
)
from qml_essentials.tape import recording
from qml_essentials.drawing import draw_text, draw_mpl, draw_tikz


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
    n_qubits: int,
    keep: List[int],
) -> jnp.ndarray:
    """Marginalize a single probability vector (no batch dimension)."""
    probs_t = probs.reshape((2,) * n_qubits)

    trace_out = sorted(set(range(n_qubits)) - set(keep), reverse=True)
    for q in trace_out:
        probs_t = probs_t.sum(axis=q)

    return probs_t.ravel()


def marginalize_probs(
    probs: jnp.ndarray,
    n_qubits: int,
    keep: List[int],
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
    if probs.shape == (dim,):
        return _marginalize_probs_single(probs, n_qubits, keep)
    # Batched: shape (B, dim)
    return jax.vmap(lambda p: _marginalize_probs_single(p, n_qubits, keep))(probs)


def build_parity_observable(
    qubit_group: List[int],
) -> Hermitian:
    """Build a multi-qubit Z⊗Z⊗...⊗Z parity observable.

    For a group of qubit indices ``[i, j, ...]``, constructs the tensor product
    ``Z_i ⊗ Z_j ⊗ ...`` as a ``Hermitian`` operation.

    Args:
        qubit_group: List of qubit indices for the parity measurement.

    Returns:
        A :class:`Hermitian` operation whose matrix is the Z parity
        tensor product and whose wires match the given qubits.
    """
    Z = PauliZ._matrix
    mat = reduce(jnp.kron, [Z] * len(qubit_group))
    return Hermitian(matrix=mat, wires=qubit_group, record=False)


# ===================================================================
# evolve – Hamiltonian time evolution (static & time-dependent)
# ===================================================================


def evolve(hamiltonian, **odeint_kwargs):
    """Return a gate-factory for Hamiltonian time evolution.

    Supports two modes:

    **Static** — when *hamiltonian* is a :class:`Hermitian`::

        gate = evolve(Hermitian(H_mat, wires=0))
        gate(t=0.5)            # U = exp(-i·0.5·H)

    **Time-dependent** — when *hamiltonian* is a
    :class:`ParametrizedHamiltonian` (created via ``coeff_fn * Hermitian``)::

        H_td = coeff_fn * Hermitian(H_mat, wires=0)
        gate = evolve(H_td)
        gate([A, sigma], T)    # U via ODE: dU/dt = -i f(p,t) H · U

    The time-dependent case solves the Schrödinger equation numerically
    using ``diffrax.diffeqsolve`` with a Dopri5 adaptive Runge-Kutta
    solver, matching PennyLane's ``ParametrizedEvolution`` implementation.

    All computations are pure JAX and fully differentiable with
    ``jax.grad``.

    Args:
        hamiltonian: Either a :class:`Hermitian` (static evolution) or a
            :class:`ParametrizedHamiltonian` (time-dependent evolution).
        **odeint_kwargs: Extra keyword arguments.  Recognised keys:

            * ``atol``, ``rtol`` — absolute/relative tolerances for the
              adaptive step-size controller (default ``1.4e-8``).

    Returns:
        A callable gate factory.  Signature depends on the mode:

        * Static: ``(t, wires=0) -> Operation``
        * Time-dependent: ``(coeff_args, T) -> Operation``

    Raises:
        TypeError: If *hamiltonian* is neither ``Hermitian`` nor
            ``ParametrizedHamiltonian``.
    """
    if isinstance(hamiltonian, Hermitian):
        return _evolve_static(hamiltonian)
    elif isinstance(hamiltonian, ParametrizedHamiltonian):
        return _evolve_parametrized(hamiltonian, **odeint_kwargs)
    else:
        raise TypeError(
            f"evolve() expects a Hermitian or ParametrizedHamiltonian, "
            f"got {type(hamiltonian)}"
        )


def _evolve_static(hermitian: Hermitian) -> Callable:
    """Gate factory for static Hamiltonian evolution U = exp(-i t H)."""
    H_mat = hermitian.matrix

    def _apply(t: float, wires: Union[int, List[int]] = 0) -> Operation:
        U = jax.scipy.linalg.expm(-1j * t * H_mat)
        return Operation(wires=wires, matrix=U)

    return _apply


def _evolve_parametrized(ph: ParametrizedHamiltonian, **odeint_kwargs) -> Callable:
    """Gate factory for time-dependent Hamiltonian evolution.

    Solves the matrix ODE ``dU/dt = -i f(params, t) H · U`` with
    ``U(0) = I`` using ``diffrax.diffeqsolve`` (Dopri5 adaptive RK),
    matching PennyLane's ``ParametrizedEvolution``.

    Performance improvements over the previous ``jax.experimental.ode``
    implementation:

    * Uses **diffrax** — a modern, well-maintained JAX ODE library with
      better XLA compilation, adjoint methods, and step-size control.
    * The ODE solve is wrapped in ``jax.jit`` so repeated calls with
      different parameters reuse the compiled XLA computation.
    * Pre-computes ``-i·H`` once instead of multiplying at every RHS
      evaluation.
    * Avoids dynamic ``jnp.where`` branching for the time span.

    Args:
        ph: A :class:`ParametrizedHamiltonian` holding the coefficient
            function, the Hamiltonian matrix, and wire indices.
        **odeint_kwargs: ``atol`` and ``rtol`` for the step-size controller.
    """
    H_mat = ph.H_mat
    coeff_fn = ph.coeff_fn
    wires = ph.wires
    dim = H_mat.shape[0]

    # Pre-compute -i·H once (avoids repeated multiplication in RHS)
    neg_iH = -1j * H_mat

    # Extract tolerances, default to PennyLane values
    atol = odeint_kwargs.pop("atol", 1.4e-8)
    rtol = odeint_kwargs.pop("rtol", 1.4e-8)

    # Pre-build solver and step-size controller (stateless, reusable)
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(atol=atol, rtol=rtol)

    # JIT-compiled ODE solve kernel — the core performance win.
    # This is traced once and reused for every call with different params/T.
    @jax.jit
    def _solve(params, t0, t1):
        """Solve dU/dt = f(params,t) · (-iH) · U from t0 to t1."""

        def rhs(t, y, args):
            return coeff_fn(args, t) * (neg_iH @ y)

        y0 = jnp.eye(dim, dtype=jnp.complex128)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            solver,
            t0=t0,
            t1=t1,
            dt0=None,  # let the controller choose the initial step
            y0=y0,
            args=params,
            stepsize_controller=stepsize_controller,
            max_steps=4096,
        )

        # sol.ys has shape (1, dim, dim) for SaveAt(t1=True) (default)
        return sol.ys[0]

    def _apply(coeff_args, T) -> Operation:
        """Evolve under the time-dependent Hamiltonian.

        Args:
            coeff_args: List of parameter sets, one per Hamiltonian term.
                Following PennyLane convention, ``coeff_args[0]`` is
                forwarded to ``coeff_fn(params, t)`` as the first argument.
            T: Total evolution time.  If scalar, the ODE is solved on
                ``[0, T]``.  If a 2-element array, on ``[T[0], T[1]]``.

        Returns:
            An :class:`Operation` wrapping the computed unitary.
        """
        # PennyLane convention: coeff_args is a list of param-sets,
        # one per term.  Single term → unpack the first.
        params = coeff_args[0] if isinstance(coeff_args, (list, tuple)) else coeff_args

        # Build time span — resolve at Python level to avoid traced branching
        T_arr = jnp.asarray(T, dtype=jnp.float64)
        if T_arr.ndim == 0:
            t0 = jnp.float64(0.0)
            t1 = T_arr
        else:
            t0 = T_arr[0]
            t1 = T_arr[1]

        U = _solve(params, t0, t1)

        return Operation(wires=wires, matrix=U)

    return _apply


# ===================================================================
# QuantumScript – circuit container & executor
# ===================================================================
class QuantumScript:
    """Circuit container and executor backed by pure JAX kernels.

    ``QuantumScript`` takes a callable *f* representing a quantum circuit.
    Within *f*, :class:`~qml_essentials.operations.Operation` objects are
    instantiated and automatically recorded onto a tape.  The tape is then
    simulated using either a statevector or density-matrix kernel depending on
    whether noise channels are present.

    Attributes:
        f: The circuit function whose body instantiates ``Operation`` objects.
        _n_qubits: Optionally pre-declared number of qubits.  When ``None``
            the qubit count is inferred from the operations recorded on the
            tape.

    Example:
        >>> def circuit(theta):
        ...     RX(theta, wires=0)
        ...     PauliZ(wires=1)
        >>> script = QuantumScript(circuit, n_qubits=2)
        >>> result = script.execute(type="expval", obs=[PauliZ(0)])
    """

    def __init__(self, f: Callable, n_qubits: Optional[int] = None) -> None:
        """Initialise a QuantumScript.

        Args:
            f: A function whose body instantiates ``Operation`` objects.
                Signature: ``f(*args, **kwargs) -> None``.
            n_qubits: Number of qubits.  If ``None``, inferred from the
                operations recorded on the tape.
        """
        self.f = f
        self._n_qubits = n_qubits

    # -- internal: record the tape by running f -------------------------
    def _record(self, *args, **kwargs) -> List[Operation]:
        """Run the circuit function and collect the recorded operations.

        Uses :func:`~qml_essentials.tape.recording` as a context manager so
        that the tape is always cleaned up — even if the circuit function
        raises — and nested recordings (e.g. from ``_execute_batched``) each
        get their own independent tape.

        Args:
            *args: Positional arguments forwarded to the circuit function.
            **kwargs: Keyword arguments forwarded to the circuit function.

        Returns:
            List of :class:`~qml_essentials.operations.Operation` instances in
            the order they were instantiated.
        """
        with recording() as tape:
            self.f(*args, **kwargs)
        return tape

    # -- infer qubit count from the tape --------------------------------
    @staticmethod
    def _infer_n_qubits(ops: List[Operation], obs: List[Operation]) -> int:
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

    # ------------------------------------------------------------------
    # Pure simulation kernels
    #
    # Both are static, accept only JAX arrays + plain Python lists, and
    # have no side-effects after the tape is built.  This makes them safe
    # targets for jax.jit, jax.grad, jax.vmap, and — in the future —
    # jax.shard_map for multi-device parallelism.
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_pure(tape: List[Operation], n_qubits: int) -> jnp.ndarray:
        """Statevector simulation kernel.

        Starts from |00…0⟩ and applies each gate in *tape* via
        :meth:`~qml_essentials.operations.Operation.apply_to_state`.

        Args:
            tape: Ordered list of gate operations to apply.
            n_qubits: Total number of qubits.

        Returns:
            Statevector of shape ``(2**n_qubits,)``.
        """
        dim = 2**n_qubits
        state = jnp.zeros(dim, dtype=jnp.complex128).at[0].set(1.0)
        for op in tape:
            state = op.apply_to_state(state, n_qubits)
        return state

    @staticmethod
    def _simulate_mixed(tape: List[Operation], n_qubits: int) -> jnp.ndarray:
        """Density-matrix simulation kernel.

        Starts from ρ = |00…0⟩⟨00…0| and applies each gate in *tape* via
        :meth:`~qml_essentials.operations.Operation.apply_to_density`
        (ρ → UρU† for unitaries, Σ_k K_k ρ K_k† for Kraus channels).
        Required for noisy circuits.

        Args:
            tape: Ordered list of gate or channel operations to apply.
            n_qubits: Total number of qubits.

        Returns:
            Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        dim = 2**n_qubits
        rho = jnp.zeros((dim, dim), dtype=jnp.complex128).at[0, 0].set(1.0)
        for op in tape:
            rho = op.apply_to_density(rho, n_qubits)
        return rho

    # -- unified simulate-and-measure ------------------------------------

    @staticmethod
    def _simulate_and_measure(
        tape: List[Operation],
        n_qubits: int,
        type: str,
        obs: List[Operation],
        use_density: bool,
    ) -> jnp.ndarray:
        """Run simulation and measurement in a single dispatch.

        Chooses statevector or density-matrix simulation based on
        *use_density*, then applies the appropriate measurement function.
        This eliminates duplicated branching logic in single-sample and
        batched execution paths.

        Args:
            tape: Ordered list of gate/channel operations to apply.
            n_qubits: Total number of qubits.
            type: Measurement type (``"state"``/``"probs"``/``"expval"``/
                ``"density"``).
            obs: Observables for ``"expval"`` measurements.
            use_density: If ``True``, use density-matrix simulation.

        Returns:
            Measurement result (shape depends on *type*).
        """
        if use_density:
            rho = QuantumScript._simulate_mixed(tape, n_qubits)
            return QuantumScript._measure_density(rho, n_qubits, type, obs)
        state = QuantumScript._simulate_pure(tape, n_qubits)
        return QuantumScript._measure_state(state, n_qubits, type, obs)

    # -- measurement helpers --------------------------------------------

    @staticmethod
    def _measure_state(
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

            * ``"state"``  → ``(2**n_qubits,)``
            * ``"probs"``  → ``(2**n_qubits,)``
            * ``"expval"`` → ``(len(obs),)``

        Raises:
            ValueError: If *type* is not a recognised measurement type.
        """
        if type == "state":
            return state

        if type == "probs":
            return jnp.abs(state) ** 2

        if type == "expval":
            # ⟨ψ|O|ψ⟩ = Re(⟨ψ| (O|ψ⟩))  — avoids building the full matrix
            return jnp.array(
                [
                    jnp.real(jnp.vdot(state, ob.apply_to_state(state, n_qubits)))
                    for ob in obs
                ]
            )

        raise ValueError(f"Unknown measurement type: {type!r}")

    @staticmethod
    def _measure_density(
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

            * ``"density"`` → ``(2**n_qubits, 2**n_qubits)``
            * ``"probs"``   → ``(2**n_qubits,)``
            * ``"expval"``  → ``(len(obs),)``

        Raises:
            ValueError: If *type* is ``"state"`` (not valid for mixed circuits)
                or another unrecognised type.
        """
        if type == "density":
            return rho

        if type == "probs":
            return jnp.real(jnp.diag(rho))

        if type == "expval":
            # Tr(O ρ) = Σ_ij O_ij ρ_ji = Σ_ij O_ij (ρ^T)_ij
            # Using elementwise multiply + sum avoids materializing O @ ρ.
            rho_T = rho.T
            return jnp.array(
                [jnp.real(jnp.sum(ob.lifted_matrix(n_qubits) * rho_T)) for ob in obs]
            )

        raise ValueError(
            "Measurement type 'state' is not defined for mixed (noisy) circuits. "
            "Use 'density' instead."
        )

    # -- core execution -------------------------------------------------

    def execute(
        self,
        type: str = "expval",
        obs: Optional[List[Operation]] = None,
        *,
        args: tuple = (),
        kwargs: dict = {},
        in_axes: Optional[Tuple] = None,
    ) -> jnp.ndarray:
        """Execute the circuit and return measurement results.

        Args:
            type: Measurement type.  One of:

                * ``"expval"``  — expectation value ⟨ψ|O|ψ⟩ / Tr(Oρ) for
                  each observable in *obs*.
                * ``"probs"``   — probability vector of shape ``(2**n,)``.
                * ``"state"``   — raw statevector of shape ``(2**n,)``.
                * ``"density"`` — full density matrix of shape
                  ``(2**n, 2**n)``.

            obs: Observables required when *type* is ``"expval"``.
            args: Positional arguments forwarded to the circuit function *f*.
            kwargs: Keyword arguments forwarded to *f*.
            in_axes: Batch axes for each element of *args*, following the same
                convention as ``jax.vmap``:

                * An integer selects that axis of the corresponding array as
                  the batch dimension.
                * ``None`` broadcasts the argument (no batching).

                When provided, :meth:`execute` calls ``jax.vmap`` over the
                pure simulation kernel and returns results with a leading
                batch dimension.

                Example — batch over axis 0 of a parameter array::

                    script.execute(
                        type="expval",
                        obs=[PauliZ(0)],
                        args=(thetas,),
                        in_axes=(0,),
                    )

                Example — batch over axis 2 of params, axis 0 of inputs::

                    script.execute(
                        type="expval",
                        obs=[PauliZ(0)],
                        args=(params, inputs),
                        in_axes=(2, 0),
                    )

        Returns:
            Without *in_axes*: shape determined by *type*.
            With *in_axes*: shape ``(B, ...)`` with a leading batch dimension.

        Note:
            **Tape / kernel split** — the circuit function is executed in
            Python *once* to record the tape and determine ``n_qubits`` and
            whether noise is present.  The pure JAX kernels
            (``_simulate_pure`` / ``_simulate_mixed``) are then vmapped, so
            Python overhead is O(circuit_depth), not O(B × circuit_depth).

            **shard_map migration** — the ``jax.vmap`` call in
            :meth:`_execute_batched` is the exact boundary to replace with
            ``jax.shard_map`` for multi-device execution.
        """
        if obs is None:
            obs = []

        if in_axes is not None:
            return self._execute_batched(
                type=type, obs=obs, args=args, kwargs=kwargs, in_axes=in_axes
            )

        # ------------------------------------------------------------------
        # Single-sample path
        # ------------------------------------------------------------------
        tape = self._record(*args, **kwargs)
        n_qubits = self._n_qubits or self._infer_n_qubits(tape, obs)

        has_noise = any(isinstance(op, KrausChannel) for op in tape)
        use_density = type == "density" or has_noise

        return self._simulate_and_measure(tape, n_qubits, type, obs, use_density)

    # -- batched execution ----------------------------------------------

    def _execute_batched(
        self,
        type: str,
        obs: List[Operation],
        args: tuple,
        kwargs: dict,
        in_axes: Tuple,
    ) -> jnp.ndarray:
        """Vectorise :meth:`execute` over a batch axis using ``jax.vmap``.

        The circuit function is traced **once** in Python with scalar slices to
        record the tape, determine ``n_qubits``, and detect noise.  The
        resulting pure simulation kernel is then vmapped over the requested
        axes.

        Args:
            type: Measurement type (see :meth:`execute`).
            obs: Observables (see :meth:`execute`).
            args: Positional arguments for the circuit function.
            kwargs: Keyword arguments for the circuit function.
            in_axes: One entry per element of *args*.  Follows ``jax.vmap``
                convention: an int gives the batch axis; ``None`` broadcasts.

        Returns:
            Batched measurement results of shape ``(B, ...)`` where *B* is the
            size of the batch dimension.

        Raises:
            ValueError: If ``len(in_axes) != len(args)``.

        Note:
            The ``jax.vmap`` call at the end of this method is the exact
            boundary to replace with ``jax.shard_map`` for multi-device
            execution::

                from jax.sharding import PartitionSpec as P, Mesh
                result = jax.shard_map(
                    _single_execute, mesh=mesh,
                    in_specs=tuple(P(0) if ax is not None else P() for ax in in_axes),
                    out_specs=P(0),
                )(*args)
        """
        if len(in_axes) != len(args):
            raise ValueError(
                f"in_axes has {len(in_axes)} entries but args has {len(args)}. "
                "Provide one in_axes entry per positional argument."
            )

        # Step 1 — record the tape once using scalar slices of each arg.
        # This determines n_qubits and whether noise channels are present
        # without running the full batch.
        # NOTE: We use lax.index_in_dim instead of jnp.take because JAX
        # random key arrays do not support jnp.take.
        def _slice_first(a, ax):
            """Take the first element along axis *ax*."""
            return jax.lax.index_in_dim(a, 0, axis=ax, keepdims=False)

        scalar_args = tuple(
            _slice_first(a, ax) if ax is not None else a for a, ax in zip(args, in_axes)
        )
        tape = self._record(*scalar_args, **kwargs)
        n_qubits = self._n_qubits or self._infer_n_qubits(tape, obs)
        has_noise = any(isinstance(op, KrausChannel) for op in tape)
        use_density = type == "density" or has_noise

        # Step 2 — define a pure single-sample function to vmap over.
        # Re-recording inside this function is necessary: the tape may
        # contain operations whose matrices depend on the batched argument
        # (e.g. RX(theta) where theta is a JAX tracer).  jax.vmap traces
        # this function once and generates a single XLA computation for
        # the entire batch.
        def _single_execute(*single_args):
            single_tape = self._record(*single_args, **kwargs)
            return self._simulate_and_measure(
                single_tape, n_qubits, type, obs, use_density
            )

        # Step 3 — vmap over the requested axes.
        #
        # Note on JIT: we intentionally do NOT wrap this in jax.jit here.
        # Circuit functions (e.g. Model._variational) commonly use
        # Python-level control flow that depends on array values
        # (``if data_reupload[q, idx]: ...``).  This is fine under vmap
        # (which keeps concrete values), but jit turns all inputs into
        # abstract tracers and raises TracerBoolConversionError.
        #
        # Users whose circuit functions are JIT-compatible can opt in by
        # wrapping the execute call themselves::
        #
        #     jax.jit(lambda args: script.execute(..., args=args, in_axes=...))
        #
        # The vmap alone already produces a single fused XLA computation
        # for the batch — the main remaining overhead is Python-level tape
        # re-recording during tracing, which happens once per vmap call.
        return jax.vmap(_single_execute, in_axes=in_axes)(*args)

    # -- circuit drawing ------------------------------------------------

    def draw(
        self,
        figure: str = "text",
        args: tuple = (),
        kwargs: dict = {},
        **draw_kwargs: Any,
    ) -> Union[str, Any]:
        """Draw the quantum circuit.

        Records the tape by calling the circuit function with the given
        arguments, then renders the resulting gate sequence.

        Args:
            figure: Rendering backend.  One of:

                * ``"text"``  — ASCII art (returned as a ``str``).
                * ``"mpl"``   — Matplotlib figure (returns ``(fig, ax)``).
                * ``"tikz"``  — LaTeX/TikZ code via ``quantikz``
                  (returns a :class:`~qml_essentials.utils.QuanTikz.TikzFigure`).

            args: Positional arguments forwarded to the circuit function
                to record the tape.
            kwargs: Keyword arguments forwarded to the circuit function.
            **draw_kwargs: Extra options forwarded to the rendering backend:

                * ``gate_values`` (bool): Show numeric gate angles instead of
                  symbolic θ_i labels.  Default ``False``.
                * ``inputs_symbols`` (str | list): Symbol(s) used for input
                  gates.  Default ``"x"``.

        Returns:
            Depends on *figure*:

            * ``"text"``  → ``str``
            * ``"mpl"``   → ``(matplotlib.figure.Figure, matplotlib.axes.Axes)``
            * ``"tikz"``  → :class:`QuanTikz.TikzFigure`

        Raises:
            ValueError: If *figure* is not one of the supported modes.
        """
        if figure not in ("text", "mpl", "tikz"):
            raise ValueError(
                f"Invalid figure mode: {figure!r}. " "Must be 'text', 'mpl', or 'tikz'."
            )

        tape = self._record(*args, **kwargs)
        n_qubits = self._n_qubits or self._infer_n_qubits(tape, [])

        # Filter out noise channels for drawing — they clutter the diagram
        ops = [op for op in tape if not isinstance(op, KrausChannel)]

        if figure == "text":
            return draw_text(ops, n_qubits)
        elif figure == "mpl":
            return draw_mpl(ops, n_qubits, **draw_kwargs)
        else:  # tikz
            return draw_tikz(ops, n_qubits, **draw_kwargs)
