"""Memory estimation and memory-aware batch chunking.

These helpers let :class:`~qml_essentials.script.Script` decide whether a batched
simulation fits in available RAM and, if not, split it into chunks that do.  They
are pure functions (the estimates are plain Python arithmetic) so they add
essentially zero overhead when the full batch fits.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

import logging

log = logging.getLogger(__name__)

# Whether to call ``jax.clear_caches()`` between memory-aware chunks in
# :func:`execute_chunked`.  Default ``False``: clearing caches between chunks
# forces XLA to recompile the same batched program for every chunk, which is a
# major performance hit when many chunks are needed.  Set ``True`` only if you
# observe OOM growth across chunks.
CLEAR_CACHES_BETWEEN_CHUNKS: bool = False


def _element_sizes() -> Tuple[int, int]:
    """Return ``(complex_elem, real_elem)`` byte sizes for the active JAX dtype.

    JAX silently truncates complex128 to complex64 (and float64 to float32) when
    x64 mode is disabled (the default), halving memory usage.
    """
    elem = 16 if jax.config.x64_enabled else 8  # complex128 vs complex64
    return elem, elem // 2  # (complex, real/float)


def _output_bytes(
    type: str,
    batch_size: int,
    dim: int,
    elem: int,
    real_elem: int,
    n_obs: int,
) -> int:
    """Bytes of the returned/accumulated ``(batch_size, ...)`` measurement array."""
    if type == "density":
        return batch_size * dim * dim * elem
    if type == "expval":
        return batch_size * max(n_obs, 1) * real_elem
    if type == "probs":
        return batch_size * dim * real_elem
    return batch_size * dim * elem  # state


def estimate_peak_bytes(
    n_qubits: int,
    batch_size: int,
    type: str,
    use_density: bool,
    n_obs: int = 0,
    n_ops: int = 1,
) -> int:
    """Estimate peak memory (bytes) for a batched simulation.

    The estimate accounts for:

    - The batched statevector (always needed, even for density).
    - The batched output tensor (state / probs / density / expval).
    - Gate-tensor temporaries (the einsum buffers).  XLA frequently
      keeps several per-gate ``(B, dim)`` (or ``(B, dim, dim)`` for
      density) buffers alive simultaneously when fusion is not
      possible, so we multiply the per-element gate cost by *n_ops*
      (the number of operations on the recorded tape).

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
        n_ops: Number of operations on the circuit tape.  Used to
            scale the per-gate intermediate buffers.  Defaults to 1
            (backwards-compatible single-buffer estimate).

    Returns:
        Estimated peak memory in bytes.
    """
    dim = 2**n_qubits
    # Detect actual element size: JAX silently truncates complex128
    # to complex64 when x64 mode is disabled (the default).
    elem, real_elem = _element_sizes()

    # Clamp n_ops to at least 1 so callers that omit the argument
    # reproduce the previous behaviour.
    n_ops = max(int(n_ops), 1)

    # Statevector: always allocated during simulation
    sv_bytes = batch_size * dim * elem

    # Simulation intermediate: when density-matrix simulation is used,
    # the full rho (dim × dim) must be held during gate evolution —
    # even if the final output is only probs or expval.
    # apply_to_density contracts both U and U* against rho, so at least
    # two intermediate (dim × dim) buffers are alive simultaneously
    # *per applied operation*.
    if use_density:
        sim_bytes = 2 * n_ops * batch_size * dim * dim * elem
    else:
        sim_bytes = 0  # statevector is already counted above

    # Output tensor: this is the *returned* array, not the simulation
    # intermediate.  For probs/expval with density simulation the
    # density matrix is reduced to a small output *before* returning,
    # so only the reduced output coexists with the next chunk.
    out_bytes = _output_bytes(type, batch_size, dim, elem, real_elem, n_obs)

    # Gate temporaries: einsum creates a ``(B, dim)`` (statevector) or
    # ``(B, dim, dim)`` (density) buffer per gate, and XLA cannot
    # always free them between consecutive ops, so scale by ``n_ops``.
    if use_density:
        gate_tmp = n_ops * batch_size * dim * dim * elem
    else:
        gate_tmp = n_ops * batch_size * dim * elem

    # Peak = max(simulation phase, output phase).  During simulation
    # the intermediate + statevector + gate temps are alive.  After
    # measurement, only the output survives.  So peak is whichever
    # phase is larger.
    sim_peak = sv_bytes + sim_bytes + gate_tmp
    out_peak = out_bytes
    raw = max(sim_peak, out_peak)

    # 1.5× safety factor for XLA compiler temporaries, padding, etc.
    return int(raw * 1.5)


def available_memory_bytes() -> int:
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

    log.debug(f"Available memory: {mem / 1024**3:.1f} GB")
    return mem


def compute_chunk_size(
    n_qubits: int,
    batch_size: int,
    type: str,
    use_density: bool,
    n_obs: int = 0,
    memory_fraction: float = 0.8,
    n_ops: int = 1,
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
        n_ops: Number of operations on the recorded tape.  Forwarded
            to :func:`estimate_peak_bytes`.  Defaults to 1.

    Returns:
        Chunk size (number of batch elements per sub-batch).
    """
    avail = int(available_memory_bytes() * memory_fraction)
    full_est = estimate_peak_bytes(
        n_qubits, batch_size, type, use_density, n_obs, n_ops=n_ops
    )

    if full_est <= avail:
        return batch_size  # everything fits — no chunking

    # The output accumulator (the final (batch_size, ...) result array)
    # must coexist with each chunk's computation, so subtract its size
    # from available memory before sizing chunks.
    dim = 2**n_qubits
    elem, real_elem = _element_sizes()
    accum_bytes = _output_bytes(type, batch_size, dim, elem, real_elem, n_obs)
    avail_for_chunks = max(avail - accum_bytes, elem)  # at least 1 element

    # Per-element cost: the memory for computing a single batch element.
    per_elem = estimate_peak_bytes(n_qubits, 1, type, use_density, n_obs, n_ops=n_ops)

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


def execute_chunked(
    batched_fn: Callable,
    args: tuple,
    in_axes: Tuple,
    batch_size: int,
    chunk_size: int,
    clear_caches: bool = False,
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
        clear_caches: When ``True``, call ``jax.clear_caches()`` after each
            chunk to release device buffers.  Disabled by default because it
            forces full recompilation of *batched_fn* on every subsequent chunk.

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
        # Optionally trigger a JAX cache clear to release device
        # buffers — disabled by default because it forces full
        # recompilation of ``batched_fn`` on every subsequent
        # chunk.  Enable by passing ``clear_caches=True`` if you
        # actually observe OOM growth across chunks.
        if clear_caches:
            jax.clear_caches()  # TODO: confirm to remove

    return output
