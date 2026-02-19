from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from qml_essentials.operations import (
    Hermitian,
    Operation,
    KrausChannel,
)
from qml_essentials.tape import recording

# Enable 64-bit precision (important for quantum simulation accuracy)
jax.config.update("jax_enable_x64", True)


# ===================================================================
# evolve – Hamiltonian time evolution
# ===================================================================
def evolve(hermitian: Hermitian) -> Callable:
    """Return a gate-factory that applies the unitary U(t) = exp(-i t H).

    The returned callable accepts a scalar parameter ``t`` and optional
    ``wires``, and is fully differentiable through ``jax.grad`` because
    ``jax.scipy.linalg.expm`` supports reverse-mode AD.

    Args:
        hermitian: A :class:`~qml_essentials.operations.Hermitian` instance
            whose matrix is used as the Hamiltonian H.

    Returns:
        A callable ``gate_factory(t, wires=0)`` that, when called inside a
        circuit function, creates and records an ``EvolvedOp`` on the active
        tape.

    Example:
        >>> time_evol = evolve(Hermitian(matrix=sigma_z, wires=0))
        >>> time_evol(t=0.5, wires=0)  # inside a circuit
    """
    H_mat = hermitian.matrix

    def _apply(t: float, wires: Union[int, List[int]] = 0) -> Operation:
        """Create and record the time-evolved unitary gate.

        Args:
            t: Evolution time (scalar).
            wires: Qubit index or list of qubit indices this gate acts on.

        Returns:
            An ``Operation`` instance wrapping U(t) = exp(-i t H).
        """
        U = jax.scipy.linalg.expm(-1j * t * H_mat)
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
        scalar_args = tuple(
            jnp.take(a, 0, axis=ax) if ax is not None else a
            for a, ax in zip(args, in_axes)
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
