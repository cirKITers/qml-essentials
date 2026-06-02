from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np  # needed to prevent jitting some operations

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
        arguments are tracers rather than concrete arrays.  In that case the
        per-:class:`Script` wrapper cache must be bypassed: a cached function
        built for a previous trace closes over that trace's tracers and would
        leak them into the current one.  The concrete-only fast path (the
        ahead-of-time-compiled executable) is likewise invalid for tracers.
        """
        return any(
            isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(args)
        )

    def _fast_call(
        self,
        cache_key: tuple,
        batched_fn: Callable,
        plain_fn: Optional[Callable],
        args: tuple,
        batch_size: int,
    ) -> jnp.ndarray:
        """Dispatch a full (unchunked) batch through the leanest path available.

        When ``plain_fn`` is provided (every positional argument is a concrete
        array), the vmapped kernel is ahead-of-time lowered and compiled to an
        XLA executable, cached per ``(cache_key, batch_size)``, and invoked
        directly.  Calling the compiled executable skips both the per-call
        pytree partition/combine of :func:`eqx.filter_jit` and the
        just-in-time cache-key recomputation; for small circuits evaluated in
        a tight loop (e.g. training iterations) that dispatch overhead, not the
        XLA compute, dominates wall-clock time.

        Falls back to ``batched_fn`` (the ``eqx.filter_jit`` wrapper) when no
        AOT-eligible ``plain_fn`` exists — i.e. when an argument is a non-array
        Python value handled as static, or the call is under a transform.
        """
        if plain_fn is None:
            return batched_fn(*args)
        aot_key = ("_aot", cache_key, batch_size)
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

        # Determine batch size from the first batched arg
        batch_size = 1
        for a, ax in zip(args, in_axes):
            if ax is not None:
                batch_size = a.shape[ax]
                break

        arg_shapes = tuple(
            (a.shape, a.dtype) if hasattr(a, "shape") else type(a) for a in args
        )
        cache_kwargs = _make_hashable(
            {k: v for k, v in kwargs.items() if not isinstance(v, jnp.ndarray)}
        )

        # TODO: we need to fix the dirty class-level `batch_gate_error` hack
        cache_key = (
            type,
            in_axes,
            arg_shapes,
            cache_kwargs,
            UnitaryGates.batch_gate_error,
        )

        # Running under an outer JAX transform (e.g. ``jax.jacrev``) makes
        # ``args`` tracers.  A cached wrapper/executable built for a previous
        # (or concrete) trace would leak tracers if reused here, so bypass the
        # per-Script wrapper cache entirely in that case; XLA-level
        # compilation caching is unaffected.
        in_transform = self._args_contain_tracer(args)

        # --- Cache-hit fast path (no shots, concrete args) ---
        cached = self._jit_cache.get(cache_key)
        if cached is not None and shots is None and not in_transform:
            batched_fn, n_qubits, use_density, n_ops, plain_fn = cached
            # Reuse the chunk size determined for this exact batch_size
            # (avoids repeated psutil syscalls).
            mem_key = ("_mem", cache_key, batch_size)
            chunk_size = self._jit_cache.get(mem_key)
            if chunk_size is None:
                chunk_size = memory.compute_chunk_size(
                    n_qubits, batch_size, type, use_density, len(obs), n_ops=n_ops
                )
                self._jit_cache[mem_key] = chunk_size
            if chunk_size >= batch_size:
                return self._fast_call(
                    cache_key, batched_fn, plain_fn, args, batch_size
                )
            return memory.execute_chunked(
                batched_fn,
                args,
                in_axes,
                batch_size,
                chunk_size,
                clear_caches=memory.CLEAR_CACHES_BETWEEN_CHUNKS,
            )

        # Record the tape once using scalar slices of each arg.
        # This determines n_qubits and whether noise channels are present
        # without running the full batch.
        # Note, that we use lax.index_in_dim instead of jnp.take because JAX
        # random key arrays do not support jnp.take.
        # TODO: fix once that is available in JAX
        def _slice_first(a, ax):
            """Take the first element along axis *ax*."""
            return jax.lax.index_in_dim(a, 0, axis=ax, keepdims=False)

        scalar_args = tuple(
            _slice_first(a, ax) if ax is not None else a for a, ax in zip(args, in_axes)
        )
        tape = self._record(*scalar_args, **kwargs)
        n_qubits = self._n_qubits or simulation.infer_n_qubits(tape, obs)
        use_density = simulation.uses_density(tape, type)
        n_ops = len(tape)

        chunk_size = memory.compute_chunk_size(
            n_qubits, batch_size, type, use_density, len(obs), n_ops=n_ops
        )

        # Re-recording inside this function is necessary: the tape may
        # contain operations whose matrices depend on the batched argument
        # (e.g. RX(theta) where theta is a JAX tracer).  jax.vmap traces
        # this function once and generates a single XLA computation for
        # the entire batch.
        if shots is not None and type in ("probs", "expval"):
            # Shot mode: compute exact probabilities, then sample.
            # The shot key is appended as an extra vmapped argument.
            def _single_execute_shots(*single_args_and_key):
                *single_args, shot_key = single_args_and_key
                single_tape = self._record(*single_args, **kwargs)
                exact_result = simulation.simulate_and_measure(
                    single_tape, n_qubits, "probs", obs, use_density
                )
                return simulation.sample_shots(
                    exact_result, n_qubits, type, obs, shots, shot_key
                )

            shot_keys = jax.random.split(key, batch_size)
            shot_in_axes = in_axes + (0,)  # key is batched over axis 0
            shot_args = args + (shot_keys,)

            # Shot-mode uses a separate cache key (includes shots)
            shot_cache_key = (
                type,
                "shots",
                shots,
                in_axes,
                arg_shapes,
                UnitaryGates.batch_gate_error,
            )
            cached_shot = self._jit_cache.get(shot_cache_key)
            if cached_shot is not None:
                batched_fn = cached_shot[0]
            else:
                batched_fn = eqx.filter_jit(
                    jax.vmap(_single_execute_shots, in_axes=shot_in_axes)
                )
                self._jit_cache[shot_cache_key] = (
                    batched_fn,
                    n_qubits,
                    use_density,
                    n_ops,
                )

            if chunk_size >= batch_size:
                return batched_fn(*shot_args)
            return memory.execute_chunked(
                batched_fn,
                shot_args,
                shot_in_axes,
                batch_size,
                chunk_size,
                clear_caches=memory.CLEAR_CACHES_BETWEEN_CHUNKS,
            )

        def _single_execute(*single_args):
            single_tape = self._record(*single_args, **kwargs)
            return simulation.simulate_and_measure(
                single_tape, n_qubits, type, obs, use_density
            )

        # General dispatch path.  Wrapping the vmapped function in
        # eqx.filter_jit has three effects:
        # 1. Static/dynamic partitioning — non-array arguments are treated as
        #    static, so circuit signatures mixing arrays and Python values work.
        # 2. Multi-core utilisation — the JIT-compiled XLA program can use
        #    intra-op parallelism across CPU threads, unlike an eager vmap.
        # 3. Compilation caching — subsequent calls with the same input shapes
        #    reuse the compiled kernel and skip Python-level tracing.
        # NOTE: when altering properties of the model, this might not get re-compiled
        # TODO: we might want to rework the data_reupload mechanism at some point
        batched_fn = eqx.filter_jit(jax.vmap(_single_execute, in_axes=in_axes))

        # Fast lane: when every positional argument is a concrete array, the
        # same vmapped kernel can be ahead-of-time compiled and the resulting
        # XLA executable called directly (see :meth:`_fast_call`), which is
        # markedly cheaper to dispatch in a repeated-call loop.  Build the
        # plain ``jax.jit`` wrapper here; ``_fast_call`` lowers+compiles it
        # lazily per batch size.  Skipped under an outer transform, where only
        # the tracer-tolerant ``batched_fn`` is valid.
        plain_fn = None
        if not in_transform and all(
            isinstance(a, (jnp.ndarray, np.ndarray)) for a in args
        ):
            plain_fn = jax.jit(jax.vmap(_single_execute, in_axes=in_axes))

        # Cache the wrappers + metadata for subsequent calls.  Skip caching
        # under an outer transform: ``_single_execute`` closes over this
        # trace's ``n_qubits``/``obs``/``kwargs``, so reusing the wrapper in a
        # different trace would leak tracers.
        if not in_transform:
            self._jit_cache[cache_key] = (
                batched_fn,
                n_qubits,
                use_density,
                n_ops,
                plain_fn,
            )

        if chunk_size >= batch_size:
            return self._fast_call(cache_key, batched_fn, plain_fn, args, batch_size)
        return memory.execute_chunked(
            batched_fn,
            args,
            in_axes,
            batch_size,
            chunk_size,
            clear_caches=memory.CLEAR_CACHES_BETWEEN_CHUNKS,
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
