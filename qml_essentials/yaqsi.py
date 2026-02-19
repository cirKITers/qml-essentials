from __future__ import annotations

from fractions import Fraction
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

# Enable 64-bit precision (important for quantum simulation accuracy)
jax.config.update("jax_enable_x64", True)


# ===================================================================
# Measurement helpers — partial trace & marginalization
# ===================================================================


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
    mat = Z
    for _ in range(len(qubit_group) - 1):
        mat = jnp.kron(mat, Z)
    return Hermitian(matrix=mat, wires=qubit_group, record=False)


# ===================================================================
# evolve – Hamiltonian time evolution (static & time-dependent)
# ===================================================================

from jax.experimental.ode import odeint as _odeint


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
    using ``jax.experimental.ode.odeint`` (Dopri5 adaptive Runge-Kutta),
    matching PennyLane's ``ParametrizedEvolution`` implementation.

    All computations are pure JAX and fully differentiable with
    ``jax.grad``.

    Args:
        hamiltonian: Either a :class:`Hermitian` (static evolution) or a
            :class:`ParametrizedHamiltonian` (time-dependent evolution).
        **odeint_kwargs: Extra keyword arguments forwarded to
            ``jax.experimental.ode.odeint`` (e.g. ``atol``, ``rtol``).

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
        return type("EvolvedOp", (Operation,), {})(wires=wires, matrix=U)

    return _apply


def _evolve_parametrized(ph: ParametrizedHamiltonian, **odeint_kwargs) -> Callable:
    """Gate factory for time-dependent Hamiltonian evolution.

    Solves the matrix ODE ``dU/dt = -i f(params, t) H · U`` with
    ``U(0) = I`` using ``jax.experimental.ode.odeint`` (Dopri5
    adaptive RK), matching PennyLane's ``ParametrizedEvolution``.

    Args:
        ph: A :class:`ParametrizedHamiltonian` holding the coefficient
            function, the Hamiltonian matrix, and wire indices.
        **odeint_kwargs: Forwarded to ``odeint`` (``atol``, ``rtol``, …).
    """
    H_mat = ph.H_mat
    coeff_fn = ph.coeff_fn
    wires = ph.wires
    dim = H_mat.shape[0]

    # Default tolerances matching PennyLane
    odeint_kwargs.setdefault("atol", 1.4e-8)
    odeint_kwargs.setdefault("rtol", 1.4e-8)

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

        # Build time span [0, T] (or [t0, t1] if T is a sequence)
        T_arr = jnp.asarray(T, dtype=jnp.float64)
        t_span = jnp.where(
            T_arr.ndim == 0,
            jnp.stack([0.0, T_arr]),
            T_arr,
        )

        # Initial condition: U(0) = I
        y0 = jnp.eye(dim, dtype=jnp.complex128)

        # RHS: dU/dt = -i f(params, t) H · U
        def rhs(y, t):
            return -1j * coeff_fn(params, t) * (H_mat @ y)

        # Solve the ODE
        sol = _odeint(rhs, y0, t_span, **odeint_kwargs)

        # sol has shape (len(t_span), dim, dim); take the final time slice
        U = sol[-1]

        return type("EvolvedOp", (Operation,), {})(wires=wires, matrix=U)

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
    # jax.experimental.shard_map for multi-device parallelism.
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
            # Tr(O ρ) — apply O column-wise to ρ via vmap, then take trace
            return jnp.array(
                [
                    jnp.real(
                        jnp.trace(
                            jax.vmap(lambda col: ob.apply_to_state(col, n_qubits))(
                                rho.T
                            ).T
                        )
                    )
                    for ob in obs
                ]
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
            ``jax.experimental.shard_map`` for multi-device execution.
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
        if type == "density" or has_noise:
            rho = self._simulate_mixed(tape, n_qubits)
            return self._measure_density(rho, n_qubits, type, obs)

        state = self._simulate_pure(tape, n_qubits)
        return self._measure_state(state, n_qubits, type, obs)

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
            boundary to replace with ``jax.experimental.shard_map`` for
            multi-device execution::

                from jax.experimental.shard_map import shard_map
                from jax.sharding import PartitionSpec as P, Mesh
                result = shard_map(
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
            if use_density:
                rho = self._simulate_mixed(single_tape, n_qubits)
                return self._measure_density(rho, n_qubits, type, obs)
            state = self._simulate_pure(single_tape, n_qubits)
            return self._measure_state(state, n_qubits, type, obs)

        # Step 3 — vmap over the requested axes.
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


# ===================================================================
# Drawing helpers
# ===================================================================

# -- Controlled-gate detection ------------------------------------------

#: Single-qubit gate names that have controlled variants
_CONTROLLED_GATES = {"CX", "CY", "CZ", "CRX", "CRY", "CRZ", "CCX", "CNOT"}


def _is_controlled(op: Operation) -> bool:
    """Return True if *op* is a controlled gate."""
    return op.name in _CONTROLLED_GATES


def _ctrl_target_name(name: str) -> str:
    """Strip the leading 'C' from a controlled gate name to get the target name."""
    if name == "CNOT":
        return "X"
    if name == "CCX":
        return "X"
    # CRX → RX, CX → X, etc.
    return name[1:]


# -- Text drawing -------------------------------------------------------


def _format_param(val: float) -> str:
    """Format a numeric parameter for text display.

    Shows nice π-fractions when possible, otherwise 2 decimal places.
    """
    try:
        frac = Fraction(float(val) / float(jnp.pi)).limit_denominator(100)
        if frac.numerator == 0:
            return "0"
        if frac.denominator <= 12:
            if frac == Fraction(1, 1):
                return "π"
            if frac.numerator == 1:
                return f"π/{frac.denominator}"
            if frac.denominator == 1:
                return f"{frac.numerator}π"
            return f"{frac.numerator}π/{frac.denominator}"
    except (ValueError, ZeroDivisionError):
        pass
    return f"{float(val):.2f}"


def _gate_label(op: Operation) -> str:
    """Build a short label like ``RX(π/2)`` or ``H`` for a gate."""
    name = op.name
    params = op.parameters
    if params:
        param_str = ", ".join(_format_param(p) for p in params)
        return f"{name}({param_str})"
    return name


def draw_text(ops: List[Operation], n_qubits: int) -> str:
    """Render a circuit tape as an ASCII-art string.

    Args:
        ops: Ordered list of gate operations (noise channels excluded).
        n_qubits: Total number of qubits.

    Returns:
        Multi-line string with one row per qubit.
    """
    if not ops:
        lines = [f" q{q}: ───" for q in range(n_qubits)]
        return "\n".join(lines)

    # Schedule operations into time-step columns.
    # Each column is a dict mapping qubit → display string.
    columns: List[Dict[int, str]] = []
    wire_busy: Dict[int, int] = {}  # qubit → next free column index

    for op in ops:
        start = max((wire_busy.get(w, 0) for w in op.wires), default=0)

        # Ensure enough columns exist
        while len(columns) <= start:
            columns.append({})

        if _is_controlled(op) and len(op.wires) >= 2:
            ctrl_wires = op.wires[:-1]
            target_wire = op.wires[-1]
            target_name = _ctrl_target_name(op.name)

            # Build target label with parameters
            if op.parameters:
                param_str = ", ".join(_format_param(p) for p in op.parameters)
                target_label = f"{target_name}({param_str})"
            else:
                target_label = target_name

            for cw in ctrl_wires:
                columns[start][cw] = "●"
            columns[start][target_wire] = target_label

            # Mark all wires in the span as busy (for crossing wires)
            all_spanned = range(min(op.wires), max(op.wires) + 1)
            for w in all_spanned:
                wire_busy[w] = start + 1
        else:
            label = _gate_label(op)
            for w in op.wires:
                columns[start][w] = label
            for w in op.wires:
                wire_busy[w] = start + 1

    # Render the grid
    # Determine column widths
    col_widths = []
    for col in columns:
        max_w = max((len(v) for v in col.values()), default=1)
        col_widths.append(max(max_w, 1))

    lines = []
    for q in range(n_qubits):
        parts = [f" q{q}: "]
        for ci, col in enumerate(columns):
            w = col_widths[ci]
            if q in col:
                cell = col[q].center(w)
            else:
                cell = "─" * w
            parts.append(f"─┤{cell}├")
        parts.append("─")
        lines.append("".join(parts))

    return "\n".join(lines)


# -- Matplotlib drawing -------------------------------------------------


def draw_mpl(
    ops: List[Operation],
    n_qubits: int,
    **kwargs: Any,
) -> Tuple:
    """Render a circuit tape as a Matplotlib figure.

    Args:
        ops: Ordered list of gate operations (noise channels excluded).
        n_qubits: Total number of qubits.
        **kwargs: Reserved for future options.

    Returns:
        Tuple ``(fig, ax)`` — a Matplotlib ``Figure`` and ``Axes``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Schedule into columns (same logic as text)
    columns: List[Dict[int, str]] = []
    wire_busy: Dict[int, int] = {}
    ctrl_info: List[Dict[str, Any]] = []  # per-column control gate metadata

    for op in ops:
        start = max((wire_busy.get(w, 0) for w in op.wires), default=0)
        while len(columns) <= start:
            columns.append({})
            ctrl_info.append({})

        if _is_controlled(op) and len(op.wires) >= 2:
            ctrl_wires = op.wires[:-1]
            target_wire = op.wires[-1]
            target_name = _ctrl_target_name(op.name)
            if op.parameters:
                param_str = ", ".join(_format_param(p) for p in op.parameters)
                target_label = f"{target_name}({param_str})"
            else:
                target_label = target_name

            for cw in ctrl_wires:
                columns[start][cw] = "●"
            columns[start][target_wire] = target_label

            ctrl_info[start] = {
                "ctrl": ctrl_wires,
                "target": target_wire,
            }

            all_spanned = range(min(op.wires), max(op.wires) + 1)
            for w in all_spanned:
                wire_busy[w] = start + 1
        else:
            label = _gate_label(op)
            for w in op.wires:
                columns[start][w] = label
                wire_busy[w] = start + 1

    n_cols = len(columns) if columns else 1
    fig_width = max(3.0, 1.2 * (n_cols + 2))
    fig_height = max(2.0, 0.8 * n_qubits)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.5, n_cols + 0.5)
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw qubit wires
    for q in range(n_qubits):
        ax.plot([-0.3, n_cols + 0.3], [q, q], color="black", linewidth=0.8, zorder=0)
        ax.text(
            -0.5,
            q,
            f"|0⟩",
            ha="right",
            va="center",
            fontsize=10,
            fontfamily="monospace",
        )

    # Draw gates
    gate_box_h = 0.6
    gate_box_w = 0.6

    for ci, col in enumerate(columns):
        x = ci + 0.5

        # Draw control lines
        ci_meta = ctrl_info[ci] if ci < len(ctrl_info) else {}
        if ci_meta:
            all_wires = list(ci_meta["ctrl"]) + [ci_meta["target"]]
            y_min = min(all_wires)
            y_max = max(all_wires)
            ax.plot([x, x], [y_min, y_max], color="black", linewidth=1.0, zorder=1)

        for q, label in col.items():
            if label == "●":
                # Control dot
                ax.plot(x, q, "o", color="black", markersize=6, zorder=3)
            else:
                # Gate box
                fontsize = 9 if len(label) <= 6 else 7
                bw = max(gate_box_w, len(label) * 0.09 + 0.2)
                rect = mpatches.FancyBboxPatch(
                    (x - bw / 2, q - gate_box_h / 2),
                    bw,
                    gate_box_h,
                    boxstyle="round,pad=0.05",
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=2,
                )
                ax.add_patch(rect)
                ax.text(
                    x,
                    q,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    zorder=4,
                )

    fig.tight_layout()
    return fig, ax


# -- TikZ drawing -------------------------------------------------------


def _tikz_param_str(val: float, op_name: str) -> str:
    """Format a rotation angle as a LaTeX string for quantikz gates."""
    try:
        frac = Fraction(float(val) / float(jnp.pi)).limit_denominator(100)
        if frac.denominator > 12:
            return f"\\gate{{{op_name}({float(val):.2f})}}"
        if frac.denominator == 1 and frac.numerator == 1:
            return f"\\gate{{{op_name}(\\pi)}}"
        if frac.numerator == 0:
            return f"\\gate{{{op_name}(0)}}"
        if frac.denominator == 1:
            return f"\\gate{{{op_name}({frac.numerator}\\pi)}}"
        if frac.numerator == 1:
            return (
                f"\\gate{{{op_name}\\left("
                f"\\frac{{\\pi}}{{{frac.denominator}}}\\right)}}"
            )
        return (
            f"\\gate{{{op_name}\\left("
            f"\\frac{{{frac.numerator}\\pi}}{{{frac.denominator}}}"
            f"\\right)}}"
        )
    except (ValueError, ZeroDivisionError):
        return f"\\gate{{{op_name}({float(val):.2f})}}"


def draw_tikz(
    ops: List[Operation],
    n_qubits: int,
    gate_values: bool = False,
    inputs_symbols: Union[str, List[str]] = "x",
    **kwargs: Any,
) -> Any:
    """Render a circuit tape as LaTeX/TikZ ``quantikz`` code.

    Args:
        ops: Ordered list of gate operations (noise channels excluded).
        n_qubits: Total number of qubits.
        gate_values: If ``True``, show numeric angles; otherwise use
            symbolic θ_i labels.
        inputs_symbols: Symbol(s) for input-encoding gates.

    Returns:
        A :class:`~qml_essentials.utils.QuanTikz.TikzFigure` object.
    """
    from qml_essentials.utils import QuanTikz

    # Build the per-wire column structure
    circuit_tikz: List[List[str]] = [["\\lstick{\\ket{0}}"] for _ in range(n_qubits)]

    # Prepare an input symbol iterator
    if isinstance(inputs_symbols, str):
        sym_iter = cycle([inputs_symbols])
    else:
        sym_iter = cycle(inputs_symbols)

    param_index = 0

    for op in ops:
        if _is_controlled(op) and len(op.wires) == 2:
            ctrl_wire = op.wires[0]
            targ_wire = op.wires[1]
            distance = targ_wire - ctrl_wire
            target_name = _ctrl_target_name(op.name)

            # Build target cell
            if op.parameters and target_name in ("RX", "RY", "RZ"):
                if gate_values:
                    targ_cell = _tikz_param_str(float(op.parameters[0]), target_name)
                else:
                    targ_cell = f"\\gate{{{target_name}(\\theta_{{{param_index}}})}}"
                param_index += 1
            elif target_name in ("X", "Y", "Z"):
                if target_name == "X":
                    targ_cell = "\\targ{}"
                else:
                    targ_cell = "\\control{}"
            else:
                targ_cell = f"\\gate{{{target_name}}}"

            ctrl_cell = f"\\ctrl{{{distance}}}"

            # Align columns for crossing wires
            crossing = range(min(op.wires), max(op.wires) + 1)
            max_len = max(len(circuit_tikz[w]) for w in crossing)
            for w in crossing:
                circuit_tikz[w].extend(
                    "" for _ in range(max_len - len(circuit_tikz[w]))
                )

            circuit_tikz[ctrl_wire].append(ctrl_cell)
            circuit_tikz[targ_wire].append(targ_cell)

            # Pad intermediate wires
            for w in crossing:
                if w != ctrl_wire and w != targ_wire:
                    circuit_tikz[w].append("")

        elif len(op.wires) == 1:
            w = op.wires[0]
            name = op.name
            # Rename for display
            if name == "Hadamard":
                name = "H"

            if gate_values and op.parameters:
                cell = _tikz_param_str(float(op.parameters[0]), name)
            elif op.parameters:
                cell = f"\\gate{{{name}(\\theta_{{{param_index}}})}}"
                param_index += 1
            else:
                cell = f"\\gate{{{name}}}"

            circuit_tikz[w].append(cell)
        else:
            # Multi-qubit gate (>2 wires) — render as a generic box
            max_len = max(len(circuit_tikz[w]) for w in op.wires)
            for w in op.wires:
                circuit_tikz[w].extend(
                    "" for _ in range(max_len - len(circuit_tikz[w]))
                )
            label = _gate_label(op)
            for w in op.wires:
                circuit_tikz[w].append(f"\\gate{{{label}}}")

    # Equalise wire lengths
    max_len = max(len(wire) for wire in circuit_tikz)
    for wire in circuit_tikz:
        wire.extend("" for _ in range(max_len - len(wire)))

    # Build the quantikz string
    quantikz_str = ""
    for wire_idx, wire_ops in enumerate(circuit_tikz):
        for op_idx, cell in enumerate(wire_ops):
            if op_idx < len(wire_ops) - 1:
                quantikz_str += f"{cell} & "
            else:
                quantikz_str += f"{cell}"
                if wire_idx < n_qubits - 1:
                    quantikz_str += " \\\\\n"

    return QuanTikz.TikzFigure(quantikz_str)
