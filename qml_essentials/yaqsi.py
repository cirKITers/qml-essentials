from functools import reduce
from typing import Any, Callable, List, Optional, Tuple, Union
import threading

import diffrax
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np  # needed to prevent jitting some operations

from qml_essentials.operations import (
    Barrier,
    Hermitian,
    ParametrizedHamiltonian,
    Operation,
    KrausChannel,
    PauliZ,
    _einsum_subscript,
    _cdtype,
)
from qml_essentials.tape import recording, pulse_recording
from qml_essentials.drawing import draw_text, draw_mpl, draw_tikz

import logging

log = logging.getLogger(__name__)


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


class Yaqsi:
    # TODO: generally, I would like to merge this into operations or vice-versa
    # and only keep Script here.  It's not clear how to do this though.

    # Module-level cache for JIT-compiled ODE solvers.  Keyed on
    # (coeff_fn_id, dim, atol, rtol) so that all evolve() calls with the
    # same pulse shape function and matrix size share one compiled XLA
    # program.  This turns O(n_gates) JIT compilations into
    # O(n_distinct_pulse_shapes) during pulse-mode circuit building.
    _evolve_solver_cache: dict = {}
    _evolve_solver_cache_lock = threading.Lock()

    @staticmethod
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

    @classmethod
    def partial_trace(
        cls,
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
            return Yaqsi._partial_trace_single(rho, n_qubits, keep)
        # Batched: shape (B, dim, dim)
        return jax.vmap(lambda r: Yaqsi._partial_trace_single(r, n_qubits, keep))(rho)

    @staticmethod
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

    @classmethod
    def marginalize_probs(
        cls,
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

        return jax.vmap(
            lambda p: Yaqsi._marginalize_probs_single(p, target_shape, trace_out)
        )(probs.reshape(-1, dim))

    @classmethod
    def build_parity_observable(
        cls,
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

    @classmethod
    def evolve(
        cls,
        hamiltonian: Union["Hermitian", "ParametrizedHamiltonian"],
        name: Optional[str] = None,
        **odeint_kwargs: Any,
    ) -> Callable:
        """Return a gate-factory for Hamiltonian time evolution.

        Supports two modes:

        Static — when *hamiltonian* is a :class:`Hermitian`::

            gate = evolve(Hermitian(H_mat, wires=0))
            gate(t=0.5)            # U = exp(-i*0.5*H)

        Time-dependent — when *hamiltonian* is a
        :class:`ParametrizedHamiltonian` (created via ``coeff_fn * Hermitian``)::

            H_td = coeff_fn * Hermitian(H_mat, wires=0)
            gate = evolve(H_td)
            gate([A, sigma], T)    # U via ODE: dU/dt = -i f(p,t) H * U

        The time-dependent case solves the Schrödinger equation numerically
        using ``diffrax.diffeqsolve`` with a Dopri8 adaptive Runge-Kutta
        solver

        All computations are pure JAX and fully differentiable with
        ``jax.grad``.

        Args:
            hamiltonian: Either a :class:`Hermitian` (static evolution) or a
                :class:`ParametrizedHamiltonian` (time-dependent evolution).
            **odeint_kwargs: Extra keyword arguments.  Recognised keys:

                - ``atol``, ``rtol`` — absolute/relative tolerances for the
                adaptive step-size controller (default ``1.4e-8``).

        Returns:
            A callable gate factory.  Signature depends on the mode:

            - Static: ``(t, wires=0) -> Operation``
            - Time-dependent: ``(coeff_args, T) -> Operation``

        Raises:
            TypeError: If *hamiltonian* is neither ``Hermitian`` nor
                ``ParametrizedHamiltonian``.
        """
        if isinstance(hamiltonian, Hermitian):
            return cls._evolve_static(hamiltonian, name=name)
        elif isinstance(hamiltonian, ParametrizedHamiltonian):
            return cls._evolve_parametrized(hamiltonian, name=name, **odeint_kwargs)
        else:
            raise TypeError(
                f"evolve() expects a Hermitian or ParametrizedHamiltonian, "
                f"got {type(hamiltonian)}"
            )

    @staticmethod
    def _evolve_static(hermitian: Hermitian, name: Optional[str] = None) -> Callable:
        """Gate factory for static Hamiltonian evolution U = exp(-i t H)."""
        H_mat = hermitian.matrix

        def _apply(t: float, wires: Union[int, List[int]] = 0) -> Operation:
            U = jax.scipy.linalg.expm(-1j * t * H_mat)
            return Operation(wires=wires, matrix=U, name=name)

        return _apply

    @classmethod
    def _evolve_parametrized(
        cls,
        ph: ParametrizedHamiltonian,
        name: Optional[str] = None,
        **odeint_kwargs: Any,
    ) -> Callable:
        """Gate factory for time-dependent Hamiltonian evolution.

        Solves the matrix ODE ``dU/dt = -i f(params, t) H * U`` with
        ``U(0) = I`` using ``diffrax.diffeqsolve`` (Dopri8 adaptive RK).

        To avoid diffrax's experimental (and potentially incorrect) complex
        dtype path, the ODE is reformulated in real arithmetic.  Writing
        ``-iH = A + iB`` (real and imaginary parts) and ``U = U_re + i U_im``,
        the system becomes::

            d(U_re)/dt = f(p,t) * (A @ U_re - B @ U_im)
            d(U_im)/dt = f(p,t) * (A @ U_im + B @ U_re)

        The state is stacked as a ``(2, dim, dim)`` real array, integrated
        with diffrax in pure ``float64`` (or ``float32`` when x64 is
        disabled), and recombined into a complex unitary at the end.

        Performance improvements over the previous ``jax.experimental.ode``
        implementation:

        Uses diffrax — a modern, well-maintained JAX ODE library with
        better XLA compilation, adjoint methods, and step-size control.
        The JIT-compiled solver is cached per coefficient function so
        that multiple ``evolve()`` calls with the same pulse shape (but
        different Hamiltonian matrices or parameters) reuse the same
        compiled XLA program.  This avoids O(n_gates) JIT compilations
        during pulse-mode tape recording.
        Pre-computes ``-i*H`` once and splits into real/imaginary parts
        to avoid complex arithmetic inside the ODE RHS.
        Avoids dynamic ``jnp.where`` branching for the time span.

        TODO: switch backonce diffrax is stable with complex arithmetic

        Args:
            ph: A :class:`ParametrizedHamiltonian` holding the coefficient
                function, the Hamiltonian matrix, and wire indices.
            **odeint_kwargs: ``atol`` and ``rtol`` for the step-size controller.
        """
        H_mat = ph.H_mat
        coeff_fn = ph.coeff_fn
        wires = ph.wires
        dim = H_mat.shape[0]

        # Pre-compute -i*H and split into real / imaginary parts so the
        # ODE RHS uses only real arithmetic (avoids diffrax complex warning).
        neg_iH = -1j * H_mat
        neg_iH_re = jnp.real(neg_iH)  # A = Re(-iH)
        neg_iH_im = jnp.imag(neg_iH)  # B = Im(-iH)
        # Stack as (2, dim, dim): [A, B]
        neg_iH_split = jnp.stack([neg_iH_re, neg_iH_im], axis=0)

        # Real dtype matching the precision mode
        _rdtype = jnp.float64 if jax.config.x64_enabled else jnp.float32

        atol = odeint_kwargs.pop("atol", 1.4e-8)
        rtol = odeint_kwargs.pop("rtol", 1.4e-8)

        # Look up or build the JIT-compiled solver for this (coeff_fn, dim)
        # combination.  All pulse gates with the same pulse shape function
        # (e.g. all RX gates share Sx, all RY share Sy) reuse a single
        # compiled XLA program, with neg_iH_split passed as a regular
        # argument rather than captured in the closure.  This turns
        # O(n_gates) JIT compilations into O(n_distinct_pulse_shapes)
        # compilations.
        #
        # We key on the function's __code__ object rather than id(coeff_fn)
        # because each pulse gate call creates a new closure (capturing the
        # rotation angle `w`), but the underlying code is identical.  Under
        # JIT, the captured `w` is a tracer anyway, so the compiled program
        # is generic over `w`.
        cache_key = (id(coeff_fn.__code__), dim, atol, rtol)

        with cls._evolve_solver_cache_lock:
            _solve = cls._evolve_solver_cache.get(cache_key)

        if _solve is None:
            solver = diffrax.Dopri8()
            stepsize_controller = diffrax.PIDController(atol=atol, rtol=rtol)

            @jax.jit
            def _solve(neg_iH_split, params, t0, t1):
                """Solve dU/dt = f(params,t) * (-iH) * U from t0 to t1.

                The state y has shape (2, dim, dim) where y[0] = Re(U)
                and y[1] = Im(U).  The Hamiltonian data neg_iH_split
                has the same layout: [Re(-iH), Im(-iH)].
                """
                A = neg_iH_split[0]  # Re(-iH)
                B = neg_iH_split[1]  # Im(-iH)

                def rhs(t, y, args):
                    c = coeff_fn(args, t)
                    u_re = y[0]
                    u_im = y[1]
                    # d(U_re)/dt = c * (A @ U_re - B @ U_im)
                    du_re = c * (A @ u_re - B @ u_im)
                    # d(U_im)/dt = c * (A @ U_im + B @ U_re)
                    du_im = c * (A @ u_im + B @ u_re)
                    return jnp.stack([du_re, du_im], axis=0)

                # Initial condition: U(0) = I  ->  Re(I) = I, Im(I) = 0
                y0 = jnp.stack(
                    [
                        jnp.eye(dim, dtype=_rdtype),
                        jnp.zeros((dim, dim), dtype=_rdtype),
                    ],
                    axis=0,
                )  # (2, dim, dim)

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

                # sol.ys has shape (1, 2, dim, dim) for SaveAt(t1=True)
                y_final = sol.ys[0]  # (2, dim, dim)
                # Recombine into complex unitary
                return y_final[0] + 1j * y_final[1]

            with cls._evolve_solver_cache_lock:
                # Double-check to avoid overwriting a concurrent build
                existing = cls._evolve_solver_cache.get(cache_key)
                if existing is not None:
                    _solve = existing
                else:
                    cls._evolve_solver_cache[cache_key] = _solve

        def _apply(coeff_args, T) -> Operation:
            """Evolve under the time-dependent Hamiltonian.

            Args:
                coeff_args: List of parameter sets, one per Hamiltonian term.
                    For static Hamiltonians, ``coeff_args[0]`` is
                    forwarded to ``coeff_fn(params, t)`` as the first argument.
                T: Total evolution time.  If scalar, the ODE is solved on
                    ``[0, T]``.  If a 2-element array, on ``[T[0], T[1]]``.

            Returns:
                An :class:`Operation` wrapping the computed unitary.
            """
            # PennyLane convention: coeff_args is a list of param-sets,
            # one per term.  Single term -> unpack the first.
            params = (
                coeff_args[0] if isinstance(coeff_args, (list, tuple)) else coeff_args
            )

            # Build time span — resolve at Python level to avoid traced branching
            T_arr = jnp.asarray(T, dtype=jnp.float64)
            if T_arr.ndim == 0:
                t0 = jnp.float64(0.0)
                t1 = T_arr
            else:
                t0 = T_arr[0]
                t1 = T_arr[1]

            U = _solve(neg_iH_split, params, t0, t1)

            return Operation(wires=wires, matrix=U, name=name)

        return _apply


# TODO adjust imports to use classmethods instead
partial_trace = Yaqsi.partial_trace
evolve = Yaqsi.evolve
marginalize_probs = Yaqsi.marginalize_probs
build_parity_observable = Yaqsi.build_parity_observable


class Script:
    """Circuit container and executor backed by pure JAX kernels.

    ``Script`` takes a callable *f* representing a quantum circuit.
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

    @staticmethod
    def _estimate_peak_bytes(
        n_qubits: int,
        batch_size: int,
        type: str,
        use_density: bool,
        n_obs: int = 0,
    ) -> int:
        """Estimate peak memory (bytes) for a batched simulation.

        The estimate accounts for:

        - The batched statevector (always needed, even for density).
        - The batched output tensor (state / probs / density / expval).
        - One gate-tensor temporary per batch element (the einsum buffer).

        Observable matrices are **not** counted: they are computed inside
        the JIT-compiled function and XLA manages their lifetime (reusing
        buffers between observables).  Similarly, the outer-product
        temporary for pure-circuit density mode is transient within XLA.

        Element size is determined dynamically from ``jax.config.x64_enabled``:
        when x64 mode is disabled (the JAX default), complex values are
        ``complex64`` (8 bytes) and floats are ``float32`` (4 bytes),
        halving memory usage compared to the x64 path.

        A 1.5× safety factor is applied to cover XLA compiler temporaries,
        padding, and other allocations not directly visible to Python.

        This is a pure Python arithmetic calculation with no JAX calls —
        it adds essentially zero overhead.

        Args:
            n_qubits: Number of qubits in the circuit.
            batch_size: Number of batch elements.
            type: Measurement type (``"state"``, ``"probs"``, ``"expval"``,
                ``"density"``).
            use_density: Whether density-matrix simulation is used.
            n_obs: Number of observables (relevant for ``"expval"``).

        Returns:
            Estimated peak memory in bytes.
        """
        dim = 2**n_qubits
        # Detect actual element size: JAX silently truncates complex128
        # to complex64 when x64 mode is disabled (the default).
        elem = 16 if jax.config.x64_enabled else 8  # complex128 vs complex64
        real_elem = elem // 2  # float64 vs float32

        # Statevector: always allocated during simulation
        sv_bytes = batch_size * dim * elem

        # Simulation intermediate: when density-matrix simulation is used,
        # the full rho (dim × dim) must be held during gate evolution —
        # even if the final output is only probs or expval.
        # apply_to_density contracts both U and U* against rho, so at least
        # two intermediate (dim × dim) buffers are alive simultaneously.
        if use_density:
            sim_bytes = 2 * batch_size * dim * dim * elem
        else:
            sim_bytes = 0  # statevector is already counted above

        # Output tensor: this is the *returned* array, not the simulation
        # intermediate.  For probs/expval with density simulation the
        # density matrix is reduced to a small output *before* returning,
        # so only the reduced output coexists with the next chunk.
        if type == "density":
            out_bytes = batch_size * dim * dim * elem
        elif type == "expval":
            out_bytes = batch_size * max(n_obs, 1) * real_elem
        elif type == "probs":
            out_bytes = batch_size * dim * real_elem
        else:  # state
            out_bytes = batch_size * dim * elem

        # Gate temporaries: einsum creates one (2,)*n buffer per batch elem
        gate_tmp = batch_size * dim * elem

        # Peak = max(simulation phase, output phase).  During simulation
        # the intermediate + statevector + gate temps are alive.  After
        # measurement, only the output survives.  So peak is whichever
        # phase is larger.
        sim_peak = sv_bytes + sim_bytes + gate_tmp
        out_peak = out_bytes
        raw = max(sim_peak, out_peak)

        # 1.5× safety factor for XLA compiler temporaries, padding, etc.
        return int(raw * 1.5)

    @staticmethod
    def _available_memory_bytes() -> int:
        """Return available system memory in bytes.

        Uses ``psutil.virtual_memory().available`` for cross-platform
        support (Linux, macOS, Windows).  Falls back to reading
        ``/proc/meminfo`` on Linux, and finally to a conservative 4 GiB
        default if neither approach succeeds.

        Returns:
            Available memory in bytes.
        """
        mem = 4 * 1024**3
        # Primary: psutil (works on Linux, macOS, Windows)
        try:
            import psutil

            mem = psutil.virtual_memory().available
        except Exception:
            log.debug("psutil not available. Fallback to /proc/meminfo")

        # Fallback: /proc/meminfo (Linux only)
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem = int(line.split()[1]) * 1024  # kB → bytes
        except Exception:
            log.debug("Failed to read /proc/meminfo. Falling back to 4 GiB")

        log.debug(f"Available memory: {mem/1024**3:.1f} GB")
        return mem

    @staticmethod
    def _compute_chunk_size(
        n_qubits: int,
        batch_size: int,
        type: str,
        use_density: bool,
        n_obs: int = 0,
        memory_fraction: float = 0.8,
    ) -> int:
        """Determine the largest chunk size that fits in available memory.

        If the full batch fits, returns *batch_size* (i.e. no chunking).
        Otherwise, returns the largest chunk size such that the computation
        of one chunk **plus** the full output accumulator fits within
        ``memory_fraction`` of available RAM.

        The output accumulator is the final ``(batch_size, ...)`` array that
        holds all results.  When chunking, this array must coexist with the
        active chunk computation, so its size is subtracted from available
        memory before computing how many elements fit per chunk.

        The minimum chunk size is 1 (fully serialised).

        Args:
            n_qubits: Number of qubits.
            batch_size: Total batch size.
            type: Measurement type.
            use_density: Whether density-matrix simulation is used.
            n_obs: Number of observables.
            memory_fraction: Fraction of available memory to target
                (default 0.8 = 80%).

        Returns:
            Chunk size (number of batch elements per sub-batch).
        """
        avail = int(Script._available_memory_bytes() * memory_fraction)
        full_est = Script._estimate_peak_bytes(
            n_qubits, batch_size, type, use_density, n_obs
        )

        if full_est <= avail:
            return batch_size  # everything fits — no chunking

        # The output accumulator (the final (batch_size, ...) result array)
        # must coexist with each chunk's computation, so subtract its size
        # from available memory before sizing chunks.
        dim = 2**n_qubits
        elem = 16 if jax.config.x64_enabled else 8
        real_elem = elem // 2
        if type == "density":
            accum_bytes = batch_size * dim * dim * elem
        elif type == "expval":
            accum_bytes = batch_size * max(n_obs, 1) * real_elem
        elif type == "probs":
            accum_bytes = batch_size * dim * real_elem
        else:
            accum_bytes = batch_size * dim * elem
        avail_for_chunks = max(avail - accum_bytes, elem)  # at least 1 element

        # Per-element cost: the memory for computing a single batch element.
        per_elem = Script._estimate_peak_bytes(n_qubits, 1, type, use_density, n_obs)

        if per_elem <= 0:
            return batch_size

        chunk = avail_for_chunks // per_elem
        chunk = max(1, min(chunk, batch_size))

        if chunk == 1 and per_elem > avail:
            log.warning(
                f"A single batch element requires ~{per_elem / 1024**3:.2f} GB "
                f"but only ~{avail / 1024**3:.2f} GB is available. "
                f"Proceeding with chunk_size=1 but OOM is possible. "
                f"Consider reducing n_qubits or switching measurement type."
            )

        log.info(
            f"Computation requires ~{full_est / 1024**3:.2f} GB which "
            f"does not fit in ~{avail / 1024**3:.2f} GB. "
            f"Using chunk size {chunk}."
        )
        return chunk

    @staticmethod
    def _execute_chunked(
        batched_fn: Callable,
        args: tuple,
        in_axes: Tuple,
        batch_size: int,
        chunk_size: int,
    ) -> jnp.ndarray:
        """Execute a vmapped function in memory-safe chunks.

        Splits the batch dimension into sub-batches of at most *chunk_size*
        elements, runs each through the JIT-compiled *batched_fn*, and
        writes results into a pre-allocated output array.

        Only one chunk's intermediate result is alive at a time: each
        chunk is computed, copied into the output buffer, and then its
        reference is dropped — allowing JAX/XLA to reclaim the memory
        before the next chunk starts.  This keeps peak memory at roughly
        ``output_buffer + one_chunk_computation`` rather than the sum of
        all chunk outputs.

        Args:
            batched_fn: A JIT-compiled, vmapped callable.
            args: Full-batch arguments (before slicing).
            in_axes: Per-argument batch axis specification.
            batch_size: Total number of batch elements.
            chunk_size: Maximum elements per chunk.

        Returns:
            Batched results with the same leading dimension as the
            full batch.
        """
        n_chunks = (batch_size + chunk_size - 1) // chunk_size
        log.debug(
            f"Memory-aware chunking: splitting batch of {batch_size} into "
            f"{n_chunks} chunks of <={chunk_size} elements."
        )

        output = None
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, batch_size)
            size = end - start

            # Slice each batched argument along its batch axis
            chunk_args = tuple(
                (
                    jax.lax.dynamic_slice_in_dim(a, start, size, axis=ax)
                    if ax is not None
                    else a
                )
                for a, ax in zip(args, in_axes)
            )

            chunk_result = batched_fn(*chunk_args)

            if output is None:
                # Pre-allocate the full output buffer on first chunk
                out_shape = (batch_size,) + chunk_result.shape[1:]
                output = jnp.zeros(out_shape, dtype=chunk_result.dtype)

            # Copy chunk into the output buffer; the slice assignment
            # creates a new array (JAX arrays are immutable) but the old
            # `output` reference is immediately replaced, letting XLA
            # reclaim it.
            output = output.at[start:end].set(chunk_result)

            # Explicitly drop the chunk reference so XLA can free the
            # chunk's device memory before computing the next one.
            del chunk_result, chunk_args
            # Trigger garbage collection to release device buffers
            jax.clear_caches()

        return output

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

    @staticmethod
    def _simulate_pure(tape: List[Operation], n_qubits: int) -> jnp.ndarray:
        """Statevector simulation kernel.

        Starts from |00…0⟩ and applies each gate in *tape* via tensor
        contraction.  The state is kept in rank-*n* tensor form ``(2,)*n``
        throughout the gate loop to avoid per-gate ``reshape`` dispatch;
        only the initial and final conversions to/from the flat ``(2**n,)``
        representation incur a reshape.

        All gate tensors and einsum subscript strings are pre-extracted from
        the tape before the simulation loop so that each iteration performs
        only a single ``jnp.einsum`` call with zero additional Python
        overhead (no method dispatch, no property access, no cache lookup).

        Args:
            tape: Ordered list of gate operations to apply.
            n_qubits: Total number of qubits.

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

        state = jnp.zeros(dim, dtype=_cdtype()).at[0].set(1.0)
        psi = state.reshape((2,) * n_qubits)
        for gt, sub in compiled:
            psi = jnp.einsum(sub, gt, psi)
        return psi.reshape(dim)

    @staticmethod
    def _simulate_mixed(tape: List[Operation], n_qubits: int) -> jnp.ndarray:
        """Density-matrix simulation kernel.

        Starts from \\rho  = \\vert 0\\rangle\\langle 0\\vert and
        applies each gate in *tape* via
        :meth:`~qml_essentials.operations.Operation.apply_to_density`
        (\\rho  -> U\\rho U† for unitaries, \\Sigma_k K_k \\rho  K_k\\dagger
        for Kraus channels).
        Required for noisy circuits.

        Args:
            tape: Ordered list of gate or channel operations to apply.
            n_qubits: Total number of qubits.

        Returns:
            Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        dim = 2**n_qubits
        rho = jnp.zeros((dim, dim), dtype=_cdtype()).at[0, 0].set(1.0)
        for op in tape:
            rho = op.apply_to_density(rho, n_qubits)
        return rho

    @staticmethod
    def _simulate_and_measure(
        tape: List[Operation],
        n_qubits: int,
        type: str,
        obs: List[Operation],
        use_density: bool,
        shots: Optional[int] = None,
        key: Optional[jnp.ndarray] = None,
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

        Returns:
            Measurement result (shape depends on *type*).
        """
        if use_density:
            # Check if any operation is actually a noise channel.
            has_noise = any(isinstance(o, KrausChannel) for o in tape)
            if has_noise:
                # Must do full density-matrix evolution for Kraus channels.
                rho = Script._simulate_mixed(tape, n_qubits)
            else:
                # Pure circuit requesting density output: simulate the
                # statevector (O(depth\times 2^n)) and form  # noqa: W605
                # \rho  = \vert\psi\rangle\langle\psi\vert once  # noqa: W605
                # (O(4^n)).  This avoids the O(depth\times 4^n) cost of  # noqa: W605
                # evolving the full density matrix gate by gate.
                state = Script._simulate_pure(tape, n_qubits)
                rho = jnp.outer(state, jnp.conj(state))

            if shots is not None and type in ("probs", "expval"):
                exact_probs = jnp.real(jnp.diag(rho))
                return Script._sample_shots(
                    exact_probs, n_qubits, type, obs, shots, key
                )
            return Script._measure_density(rho, n_qubits, type, obs)

        state = Script._simulate_pure(tape, n_qubits)

        if shots is not None and type in ("probs", "expval"):
            exact_probs = jnp.abs(state) ** 2
            return Script._sample_shots(exact_probs, n_qubits, type, obs, shots, key)
        return Script._measure_state(state, n_qubits, type, obs)

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
                    p_q = jnp.sum(
                        psi_t, axis=tuple(i for i in range(n_qubits) if i != q)
                    )
                    results.append(d[0] * p_q[0] + d[1] * p_q[1])
                return jnp.array(results)

            # General path: stack observable matrices and use a single
            # batched matmul instead of a Python loop of tensor contractions.
            # O_states[i] = obs[i] |ψ⟩, then ⟨O_i⟩ = Re(⟨ψ|O_states[i]⟩).
            obs_mats = jnp.stack(
                [ob.lifted_matrix(n_qubits) for ob in obs], axis=0
            )  # (n_obs, dim, dim)
            # Batched matvec: (n_obs, dim, dim) @ (dim,) -> (n_obs, dim)
            O_states = jnp.einsum("oij,j->oi", obs_mats, state)
            return jnp.real(jnp.einsum("i,oi->o", jnp.conj(state), O_states))

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
            obs_mats = jnp.stack(
                [ob.lifted_matrix(n_qubits) for ob in obs], axis=0
            )  # (n_obs, dim, dim)
            # einsum "oij,ji->o" computes Tr(O_o @ \\rho ) for each observable
            return jnp.real(jnp.einsum("oij,ji->o", obs_mats, rho))

        raise ValueError(
            "Measurement type 'state' is not defined for mixed (noisy) circuits. "
            "Use 'density' instead."
        )

    @staticmethod
    def _sample_shots(
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
            f"Shot simulation is only supported for 'probs' and 'expval', "
            f"got {type!r}."
        )

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

                - ``"expval"``  — expectation value ⟨ψ|O|ψ⟩ / Tr(O\\rho ) for
                  each observable in *obs*.
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
            n_qubits = self._n_qubits or self._infer_n_qubits(tape, obs)

            has_noise = any(isinstance(op, KrausChannel) for op in tape)
            use_density = type == "density" or has_noise

            return self._simulate_and_measure(
                tape,
                n_qubits,
                type,
                obs,
                use_density,
                shots=shots,
                key=key,
            )

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
        from qml_essentials.gates import UnitaryGates

        cache_key = (
            type,
            in_axes,
            arg_shapes,
            cache_kwargs,
            UnitaryGates.batch_gate_error,
        )

        # --- Cache-hit fast path (no shots) ---
        cached = self._jit_cache.get(cache_key)
        if cached is not None and shots is None:
            batched_fn, n_qubits, use_density = cached
            # Check if we already determined the chunk size for this
            # exact batch_size (avoids repeated psutil syscalls).
            mem_key = ("_mem", cache_key, batch_size)
            cached_chunk = self._jit_cache.get(mem_key)
            if cached_chunk is not None:
                if cached_chunk >= batch_size:
                    return batched_fn(*args)
                return self._execute_chunked(
                    batched_fn, args, in_axes, batch_size, cached_chunk
                )
            chunk_size = self._compute_chunk_size(
                n_qubits, batch_size, type, use_density, len(obs)
            )
            self._jit_cache[mem_key] = chunk_size
            if chunk_size >= batch_size:
                return batched_fn(*args)
            return self._execute_chunked(
                batched_fn, args, in_axes, batch_size, chunk_size
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
        n_qubits = self._n_qubits or self._infer_n_qubits(tape, obs)
        has_noise = any(isinstance(op, KrausChannel) for op in tape)
        use_density = type == "density" or has_noise

        chunk_size = self._compute_chunk_size(
            n_qubits, batch_size, type, use_density, len(obs)
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
                exact_result = self._simulate_and_measure(
                    single_tape, n_qubits, "probs", obs, use_density
                )
                return Script._sample_shots(
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
                batched_fn = jax.jit(
                    jax.vmap(_single_execute_shots, in_axes=shot_in_axes)
                )
                self._jit_cache[shot_cache_key] = (batched_fn, n_qubits, use_density)

            if chunk_size >= batch_size:
                return batched_fn(*shot_args)
            return self._execute_chunked(
                batched_fn, shot_args, shot_in_axes, batch_size, chunk_size
            )

        def _single_execute(*single_args):
            single_tape = self._record(*single_args, **kwargs)
            return self._simulate_and_measure(
                single_tape, n_qubits, type, obs, use_density
            )

        # Wrapping the vmapped function in jax.jit has two effects:
        # 1. Multi-core utilisation — the JIT-compiled XLA program can
        #    use intra-op parallelism to distribute independent SIMD lanes
        #    across CPU threads, unlike an eager vmap which runs
        #    single-threaded.
        # 2. Compilation caching — subsequent calls with the same input
        #    shapes reuse the compiled kernel and skip all Python-level
        #    tracing, eliminating the O(B\\times circuit_depth) Python overhead.
        #
        # The compiled function is cached on this Script instance,
        # keyed on (type, in_axes, arg_shapes).  Repeated calls with the
        # same structure (e.g. training iterations) skip both Python-level
        # tracing and XLA compilation entirely — they jump straight to the
        # cache check at the top of this method.
        # NOTE: when altering properties of the model, this might not get re-compiled
        # TODO: we might want to rework the data_reupload mechanism at some point
        batched_fn = jax.jit(jax.vmap(_single_execute, in_axes=in_axes))
        # Cache the function together with metadata for fast-path memory
        # checks on subsequent calls.
        self._jit_cache[cache_key] = (batched_fn, n_qubits, use_density)

        if chunk_size >= batch_size:
            return batched_fn(*args)
        return self._execute_chunked(batched_fn, args, in_axes, batch_size, chunk_size)

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
        n_qubits = self._n_qubits or self._infer_n_qubits(tape, [])

        # Filter out noise channels for drawing — they clutter the diagram
        ops = [op for op in tape if not isinstance(op, KrausChannel)]

        if figure == "text":
            return draw_text(ops, n_qubits)
        elif figure == "mpl":
            return draw_mpl(ops, n_qubits, **draw_kwargs)
        else:  # tikz
            return draw_tikz(ops, n_qubits, **draw_kwargs)
