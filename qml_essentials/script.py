from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx

from qml_essentials.operations import Operation, KrausChannel
from qml_essentials.tape import recording, pulse_recording
from qml_essentials.drawing import draw_text, draw_mpl, draw_tikz
from qml_essentials.unitary import UnitaryGates
from qml_essentials import simulation, memory


def _make_hashable(obj):
    """Recursively convert an object into a hashable form for cache keys.

    - ``dict``  → sorted tuple of ``(key, _make_hashable(value))`` pairs
    - ``list``  → tuple of ``_make_hashable(element)``
    - ``set``   → frozenset of ``_make_hashable(element)``
    - everything else is returned as-is (assumed hashable)
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, set):
        return frozenset(_make_hashable(x) for x in obj)
    return obj


class _BatchPlan(NamedTuple):
    """Compiled artefacts for one batched circuit signature.

    Cached in :attr:`Script._jit_cache` keyed on the signature ``cache_key``.
    ``batched_fn`` is deliberately the first field so callers (and tests) can
    unpack ``batched_fn, *_ = plan``.

    Attributes:
        batched_fn: ``eqx.filter_jit(jax.vmap(...))`` wrapper; always valid,
            including under an outer transform and in shot mode.
        plain_fn: AOT-eligible ``jax.jit(jax.vmap(...))`` wrapper, or ``None``
            when no concrete-array fast path applies (non-array argument, shot
            mode, or running under a transform).
        n_qubits: Qubit count derived from the recorded tape.
        use_density: Whether density-matrix simulation is required.
        n_ops: Number of operations on the tape (for memory estimation).
    """

    batched_fn: Callable
    plain_fn: Optional[Callable]
    n_qubits: int
    use_density: bool
    n_ops: int


class Script:
    """Circuit container and executor backed by pure JAX kernels.

    ``Script`` takes a callable *f* representing a quantum circuit.
    Within *f*, :class:`~qml_essentials.operations.Operation` objects are
    instantiated and automatically recorded onto a tape.  The tape is then
    simulated using either a statevector or density-matrix kernel depending on
    whether noise channels are present.

    The stateless simulation/measurement kernels live in
    :mod:`qml_essentials.simulation` and the memory-estimation/chunking helpers
    in :mod:`qml_essentials.memory`; this class orchestrates recording,
    batching, caching, and drawing around them.

    Attributes:
        f: The circuit function whose body instantiates ``Operation`` objects.
        _n_qubits: Optionally pre-declared number of qubits.  When ``None``
            the qubit count is inferred from the operations recorded on the
            tape.

    Example:
        >>> def circuit(theta):
        ...     RX(theta, wires=0)
        ...     PauliZ(wires=1)
        >>> script = Script(circuit, n_qubits=2)
        >>> result = script.execute(type="expval", obs=[PauliZ(0)])
    """

    def __init__(self, f: Callable[..., None], n_qubits: Optional[int] = None) -> None:
        """Initialise a Script.

        Args:
            f: A function whose body instantiates ``Operation`` objects.
                Signature: ``f(*args, **kwargs) -> None``.
            n_qubits: Number of qubits.  If ``None``, inferred from the
                operations recorded on the tape.
        """
        self.f = f
        self._n_qubits = n_qubits
        self._jit_cache: dict = {}  # keyed on (type, in_axes, arg_shapes, gateError)

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

    def pulse_events(self, *args, **kwargs) -> list:
        """Run the circuit and collect pulse events emitted by PulseGates.

        Activates both the normal operation tape (so gates execute) and
        a pulse-event tape that captures
        :class:`~qml_essentials.drawing.PulseEvent` objects from leaf
        pulse gates (RX, RY, RZ, CZ).

        Args:
            *args (Any): Forwarded to the circuit function.
            **kwargs (Any): Forwarded to the circuit function.

        Returns:
            List of :class:`~qml_essentials.drawing.PulseEvent`.
        """
        with pulse_recording() as events:
            with recording():
                self.f(*args, **kwargs)
        return events

    def execute(
        self,
        type: str = "expval",
        obs: Optional[List[Operation]] = None,
        *,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        in_axes: Optional[Tuple] = None,
        shots: Optional[int] = None,
        key: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Execute the circuit and return measurement results.

        Args:
            type: Measurement type.  One of:

                - ``"expval"``  — expectation value
                    \\langle\\psi|O|\\psi\\rangle / Tr(O\\rho )
                    for each observable in *obs*.
                - ``"probs"``   — probability vector of shape ``(2**n,)``.
                - ``"state"``   — raw statevector of shape ``(2**n,)``.
                - ``"density"`` — full density matrix of shape
                  ``(2**n, 2**n)``.

            obs: Observables required when type is ``"expval"``.
            args: Positional arguments forwarded to the circuit function f.
            kwargs: Keyword arguments forwarded to f.
            in_axes: Batch axes for each element of *args*, following the same
                convention as ``jax.vmap``:

                - An integer selects that axis of the corresponding array as
                  the batch dimension.
                - ``None`` broadcasts the argument (no batching).

                When provided, :meth:`execute` calls ``jax.vmap`` over the
                pure simulation kernel and returns results with a leading
                batch dimension.
            shots: Number of measurement shots for stochastic sampling.
                If ``None`` (default), exact analytic results are returned.
                Only supported for ``"probs"`` and ``"expval"`` measurement
                types.
            key: JAX PRNG key for shot sampling.  If ``None`` and *shots*
                is set, a default key ``jax.random.PRNGKey(0)`` is used.

        Returns:
            Without in_axes: shape determined by type.
            With in_axes: shape ``(B, ...)`` with a leading batch dimension.
        """
        if obs is None:
            obs = []
        if kwargs is None:
            kwargs = {}
        if shots is not None and key is None:
            key = jax.random.PRNGKey(0)

        # Split single/ parallel execution
        # TODO: we might want to unify the n_qubit stuff such that we can eliminate
        # the parameter to this method entirely
        if in_axes is not None:
            return self._execute_batched(
                type=type,
                obs=obs,
                args=args,
                kwargs=kwargs,
                in_axes=in_axes,
                shots=shots,
                key=key,
            )
        else:
            tape = self._record(*args, **kwargs)
            n_qubits = self._n_qubits or simulation.infer_n_qubits(tape, obs)

            use_density = simulation.uses_density(tape, type)

            return simulation.simulate_and_measure(
                tape,
                n_qubits,
                type,
                obs,
                use_density,
                shots=shots,
                key=key,
            )

    @staticmethod
    def _args_contain_tracer(args: tuple) -> bool:
        """Return ``True`` if any leaf of *args* is a JAX tracer.

        When :meth:`execute` runs under an outer transform (``jax.grad``,
        ``jax.jacrev``, an enclosing ``jax.jit``/``vmap``) the positional
        arguments are tracers rather than concrete arrays.  The tracer-tolerant
        ``eqx.filter_jit`` wrapper is still reused in that case (its closure
        captures only concrete metadata), but the concrete-only fast path — the
        ahead-of-time-compiled XLA executable — is invalid for tracers and must
        be skipped.
        """
        return any(
            isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(args)
        )

    @staticmethod
    def _batch_size(args: tuple, in_axes: Tuple) -> int:
        """Size of the batch dimension, read from the first batched argument."""
        for a, ax in zip(args, in_axes):
            if ax is not None:
                return a.shape[ax]
        return 1

    @staticmethod
    def _slice_first(a: Any, ax: int) -> Any:
        """Take the first element along axis *ax*.

        Uses ``jax.lax.index_in_dim`` rather than ``jnp.take`` because JAX
        random-key arrays do not support ``jnp.take``.
        """
        # TODO: fix once that is available in JAX
        return jax.lax.index_in_dim(a, 0, axis=ax, keepdims=False)

    def _record_metadata(
        self, scalar_args: tuple, kwargs: dict, obs: List[Operation], type: str
    ) -> Tuple[int, bool, int]:
        """Trace the tape from scalar slices to derive batch-invariant metadata.

        Recording once with scalar slices determines ``n_qubits`` and whether
        noise channels are present (forcing density-matrix simulation) without
        running the full batch.

        Returns:
            ``(n_qubits, use_density, n_ops)``.
        """
        tape = self._record(*scalar_args, **kwargs)
        n_qubits = self._n_qubits or simulation.infer_n_qubits(tape, obs)
        use_density = simulation.uses_density(tape, type)
        return n_qubits, use_density, len(tape)

    def _build_plan(
        self,
        type: str,
        obs: List[Operation],
        args: tuple,
        kwargs: dict,
        in_axes: Tuple,
    ) -> _BatchPlan:
        """Trace the circuit once and build the cacheable execution plan.

        Records the tape from scalar slices of *args* (to derive
        ``n_qubits``/noise), then builds the vmapped ``eqx.filter_jit``
        wrapper.  When every positional argument is array-like (so plain
        ``jax.jit`` — which has no static-argument handling — is valid) an
        AOT-eligible plain ``jax.jit`` wrapper is built too; :meth:`_dispatch`
        lowers and compiles it lazily per batch size, and only with concrete
        args (the AOT path is gated off under a transform by the caller).
        """
        scalar_args = tuple(
            self._slice_first(a, ax) if ax is not None else a
            for a, ax in zip(args, in_axes)
        )
        n_qubits, use_density, n_ops = self._record_metadata(
            scalar_args, kwargs, obs, type
        )

        # Re-recording inside this closure is necessary: tape operations may
        # have matrices that depend on the batched argument (e.g. RX(theta)
        # with theta a tracer).  jax.vmap traces this once into a single XLA
        # computation spanning the whole batch.
        def _single_execute(*single_args):
            single_tape = self._record(*single_args, **kwargs)
            return simulation.simulate_and_measure(
                single_tape, n_qubits, type, obs, use_density
            )

        # Wrapping the vmapped function in eqx.filter_jit: (1) treats non-array
        # arguments as static, so circuit signatures mixing arrays and Python
        # values work; (2) lets the XLA program use intra-op CPU parallelism;
        # (3) caches compilation across calls with the same input shapes.
        # NOTE: when altering properties of the model, this might not get
        # re-compiled.
        # TODO: we might want to rework the data_reupload mechanism at some point
        batched_fn = eqx.filter_jit(jax.vmap(_single_execute, in_axes=in_axes))

        # AOT eligibility is a structural property of the signature: plain
        # ``jax.jit`` has no static-argument handling, so it is valid only when
        # every positional argument is array-like.  ``hasattr(a, "shape")`` is
        # true for concrete arrays, numpy arrays, and tracers, but false for
        # Python statics (str/None/dict).  Building the wrapper is pure (it
        # traces nothing); the lower+compile happens lazily in :meth:`_dispatch`
        # and only with concrete args, so this is safe to build under a
        # transform — its use is gated off there by the caller.
        plain_fn = None
        if all(hasattr(a, "shape") for a in args):
            plain_fn = jax.jit(jax.vmap(_single_execute, in_axes=in_axes))

        return _BatchPlan(batched_fn, plain_fn, n_qubits, use_density, n_ops)

    def _chunk_size(
        self,
        cache_key: tuple,
        plan: _BatchPlan,
        type: str,
        n_obs: int,
        batch_size: int,
    ) -> int:
        """Largest batch chunk that fits in memory, memoized per batch size.

        The result is cached under ``("_mem", cache_key, batch_size)`` to avoid
        repeated ``psutil`` syscalls across a tight repeated-call loop.
        """
        mem_key = ("_mem", cache_key, batch_size)
        chunk_size = self._jit_cache.get(mem_key)
        if chunk_size is None:
            chunk_size = memory.compute_chunk_size(
                plan.n_qubits,
                batch_size,
                type,
                plan.use_density,
                n_obs,
                n_ops=plan.n_ops,
            )
            self._jit_cache[mem_key] = chunk_size
        return chunk_size

    def _dispatch(
        self,
        aot_key: Optional[tuple],
        batched_fn: Callable,
        plain_fn: Optional[Callable],
        args: tuple,
        in_axes: Tuple,
        batch_size: int,
        chunk_size: int,
    ) -> jnp.ndarray:
        """Run a built plan through the leanest applicable path.

        - ``chunk_size < batch_size``: the full batch would not fit in memory,
          so execute it in memory-safe sub-batches via
          :func:`~qml_essentials.memory.execute_chunked`.
        - otherwise, when an AOT-eligible ``plain_fn`` exists, ahead-of-time
          lower+compile the vmapped kernel to an XLA executable (cached per
          ``aot_key``) and call it directly.  This skips both the per-call
          pytree partition/combine of :func:`eqx.filter_jit` and its
          just-in-time cache-key recomputation; for small circuits in a tight
          loop that dispatch overhead, not the XLA compute, dominates.
        - otherwise fall back to ``batched_fn`` (no ``plain_fn``: a non-array
          argument, shot mode, or running under a transform).
        """
        if chunk_size < batch_size:
            return memory.execute_chunked(
                batched_fn,
                args,
                in_axes,
                batch_size,
                chunk_size,
                clear_caches=memory.CLEAR_CACHES_BETWEEN_CHUNKS,
            )
        if plain_fn is None:
            return batched_fn(*args)
        compiled = self._jit_cache.get(aot_key)
        if compiled is None:
            compiled = plain_fn.lower(*args).compile()
            self._jit_cache[aot_key] = compiled
        return compiled(*args)

    def _execute_batched(
        self,
        type: str,
        obs: List[Operation],
        args: tuple,
        kwargs: dict,
        in_axes: Tuple,
        shots: Optional[int] = None,
        key: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Vectorise :meth:`execute` over a batch axis using ``jax.vmap``.

        The circuit function is traced once in Python with scalar slices to
        record the tape, determine ``n_qubits``, and detect noise.  The
        resulting pure simulation kernel is then vmapped over the requested
        axes.

        Memory-aware chunking — before launching the full vmap, the
        method estimates peak memory usage.  If the full batch would exceed
        available RAM (with a safety margin), the batch is automatically
        split into sub-batches that fit.  Each chunk is vmapped independently
        and the results are concatenated.  This trades a small amount of
        wall-clock time for guaranteed execution without OOM.

        When the full batch fits in memory, there is zero overhead — the
        memory check is a pure Python arithmetic calculation (no JAX calls).

        Args:
            type: Measurement type (see :meth:`execute`).
            obs: Observables (see :meth:`execute`).
            args: Positional arguments for the circuit function.
            kwargs: Keyword arguments for the circuit function.
            in_axes: One entry per element of *args*.  Follows ``jax.vmap``
                convention: an int gives the batch axis; ``None`` broadcasts.
            shots: Number of measurement shots.  If ``None``, exact results.
            key: JAX PRNG key for shot sampling.

        Returns:
            Batched measurement results of shape ``(B, ...)`` where *B* is the
            size of the batch dimension.

        Raises:
            ValueError: If ``len(in_axes) != len(args)``.

        Note:
            The ``jax.vmap`` call in :meth:`_build_plan` is the exact
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

        batch_size = self._batch_size(args, in_axes)

        # Running under an outer JAX transform (e.g. ``jax.jacrev``) makes
        # ``args`` tracers.  The tracer-tolerant ``batched_fn`` wrapper is still
        # cached and reused (see exact-mode dispatch below); only the AOT
        # ``plain_fn`` executable is gated off, as it cannot accept tracers.
        in_transform = self._args_contain_tracer(args)

        arg_shapes = tuple(
            (a.shape, a.dtype) if hasattr(a, "shape") else type(a) for a in args
        )
        # TODO: we need to fix the dirty class-level `batch_gate_error` hack.
        # It is a global toggle that changes the compiled circuit, so it has to
        # take part in every cache key.
        gate_error = UnitaryGates.batch_gate_error

        # --- Shot mode: compute exact probabilities, then sample. ---
        if shots is not None and type in ("probs", "expval"):
            shot_cache_key = (type, "shots", shots, in_axes, arg_shapes, gate_error)
            shot_in_axes = in_axes + (0,)  # shot key batched over axis 0
            shot_args = args + (jax.random.split(key, batch_size),)

            plan = self._jit_cache.get(shot_cache_key)
            if plan is None:
                scalar_args = tuple(
                    self._slice_first(a, ax) if ax is not None else a
                    for a, ax in zip(args, in_axes)
                )
                n_qubits, use_density, n_ops = self._record_metadata(
                    scalar_args, kwargs, obs, type
                )

                # Re-recording inside the closure lets jax.vmap trace the whole
                # batch into one XLA program; the shot key is the extra vmapped
                # argument.
                def _single_execute_shots(*single_args_and_key):
                    *single_args, shot_key = single_args_and_key
                    single_tape = self._record(*single_args, **kwargs)
                    exact_result = simulation.simulate_and_measure(
                        single_tape, n_qubits, "probs", obs, use_density
                    )
                    return simulation.sample_shots(
                        exact_result, n_qubits, type, obs, shots, shot_key
                    )

                batched_fn = eqx.filter_jit(
                    jax.vmap(_single_execute_shots, in_axes=shot_in_axes)
                )
                plan = _BatchPlan(batched_fn, None, n_qubits, use_density, n_ops)
                self._jit_cache[shot_cache_key] = plan

            chunk_size = self._chunk_size(
                shot_cache_key, plan, type, len(obs), batch_size
            )
            # Shot mode never uses the AOT fast path (plain_fn is None).
            return self._dispatch(
                None,
                plan.batched_fn,
                None,
                shot_args,
                shot_in_axes,
                batch_size,
                chunk_size,
            )

        # --- Exact mode: reuse the cached plan or build it on a miss. ---
        cache_kwargs = _make_hashable(
            {k: v for k, v in kwargs.items() if not isinstance(v, jnp.ndarray)}
        )
        cache_key = (type, in_axes, arg_shapes, cache_kwargs, gate_error)

        # The cached ``batched_fn`` (eqx.filter_jit wrapper) is reused across
        # calls including under an outer transform: its ``_single_execute``
        # closure captures only concrete metadata (n_qubits/obs/use_density and
        # non-array kwargs), so it leaks no tracers, and reusing one wrapper
        # lets JAX hit its aval-keyed trace cache instead of re-tracing the
        # circuit every call.  Only the AOT ``plain_fn`` (a compiled executable)
        # is invalid for tracers; its use is gated below by ``in_transform``.
        plan = self._jit_cache.get(cache_key)
        if plan is None:
            plan = self._build_plan(type, obs, args, kwargs, in_axes)
            self._jit_cache[cache_key] = plan

        chunk_size = self._chunk_size(cache_key, plan, type, len(obs), batch_size)
        return self._dispatch(
            ("_aot", cache_key, batch_size),
            plan.batched_fn,
            None if in_transform else plan.plain_fn,
            args,
            in_axes,
            batch_size,
            chunk_size,
        )

    def draw(
        self,
        figure: str = "text",
        args: tuple = (),
        kwargs: Optional[dict] = None,
        **draw_kwargs: Any,
    ) -> Union[str, Any]:
        """Draw the quantum circuit.

        Records the tape by calling the circuit function with the given
        arguments, then renders the resulting gate sequence.

        Args:
            figure: Rendering backend.  One of:

                - ``"text"``  — ASCII art (returned as a ``str``).
                - ``"mpl"``   — Matplotlib figure (returns ``(fig, ax)``).
                - ``"tikz"``  — LaTeX/TikZ code via ``quantikz``
                  (returns a :class:`TikzFigure`).
                - ``"pulse"`` — Pulse schedule plot (returns ``(fig, axes)``).

            args: Positional arguments forwarded to the circuit function
                to record the tape.
            kwargs: Keyword arguments forwarded to the circuit function.
            **draw_kwargs: Extra options forwarded to the rendering backend:

                - ``gate_values`` (bool): Show numeric gate angles instead of
                  symbolic \\theta_i labels.  Default ``False``.
                - ``show_carrier`` (bool): For ``"pulse"`` mode, overlay the
                  carrier-modulated waveform.  Default ``False``.

        Returns:
            Depends on *figure*:

            - ``"text"``  -> ``str``
            - ``"mpl"``   -> ``(matplotlib.figure.Figure, matplotlib.axes.Axes)``
            - ``"tikz"``  -> :class:`TikzFigure`
            - ``"pulse"`` -> ``(matplotlib.figure.Figure, numpy.ndarray)``

        Raises:
            ValueError: If *figure* is not one of the supported modes.
        """
        if figure not in ("text", "mpl", "tikz", "pulse"):
            raise ValueError(
                f"Invalid figure mode: {figure!r}. "
                "Must be 'text', 'mpl', 'tikz', or 'pulse'."
            )

        if kwargs is None:
            kwargs = {}

        if figure == "pulse":
            from qml_essentials.drawing import draw_pulse_schedule

            events = self.pulse_events(*args, **kwargs)
            n_qubits = (
                self._n_qubits
                or max((w for ev in events for w in ev.wires), default=0) + 1
            )
            return draw_pulse_schedule(events, n_qubits, **draw_kwargs)

        tape = self._record(*args, **kwargs)
        n_qubits = self._n_qubits or simulation.infer_n_qubits(tape, [])

        # Filter out noise channels for drawing — they clutter the diagram
        ops = [op for op in tape if not isinstance(op, KrausChannel)]

        if figure == "text":
            return draw_text(ops, n_qubits)
        elif figure == "mpl":
            return draw_mpl(ops, n_qubits, **draw_kwargs)
        else:  # tikz
            return draw_tikz(ops, n_qubits, **draw_kwargs)
