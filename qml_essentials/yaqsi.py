from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from qml_essentials.operations import (
    _set_tape,
    Hermitian,
    Operation,
    KrausChannel,
)

# Enable 64-bit precision (important for quantum simulation accuracy)
jax.config.update("jax_enable_x64", True)


# ===================================================================
# evolve – Hamiltonian time evolution
# ===================================================================
def evolve(hermitian: Hermitian):
    """
    Returns a callable gate-factory that applies the unitary

        U(t) = exp(-i t H)

    for a given Hermitian operator *H*.  The returned callable accepts
    a scalar parameter ``t`` and optional ``wires``.

    This is fully differentiable through ``jax.grad`` because
    ``jax.scipy.linalg.expm`` supports reverse-mode AD.

    Example::

        time_evol = evolve(Hermitian(matrix=sigma_z, wires=0))
        time_evol(t=0.5, wires=0)           # inside a circuit
    """
    H_mat = hermitian.matrix

    def _apply(t: float, wires: Union[int, List[int]] = 0):
        U = jax.scipy.linalg.expm(-1j * t * H_mat)
        return type("EvolvedOp", (Operation,), {})(wires=wires, matrix=U)

    return _apply


# ===================================================================
# QuantumScript – circuit container & executor
# ===================================================================
class QuantumScript:
    """
    This forms the basis for any quantum circuit.

    It takes a callable *f* which is the circuit to execute.
    Within *f*, different operations are instantiated and automatically
    recorded onto a tape.

    Parameters
    ----------
    f : callable
        A function whose body instantiates Operation objects.
        Signature:  ``f(*args, **kwargs) -> None``
    n_qubits : int or None
        Number of qubits. If *None*, inferred from the operations.
    """

    def __init__(self, f: Callable, n_qubits: Optional[int] = None):
        self.f = f
        self._n_qubits = n_qubits

    # -- internal: record the tape by running f -------------------------
    def _record(self, *args, **kwargs) -> List[Operation]:
        tape: List[Operation] = []
        _set_tape(tape)
        try:
            self.f(*args, **kwargs)
        finally:
            _set_tape(None)
        return tape

    # -- infer qubit count from the tape --------------------------------
    @staticmethod
    def _infer_n_qubits(ops: List[Operation], obs: List[Operation]) -> int:
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
        """
        Statevector simulation kernel.

        Starts from |00…0⟩ and applies each gate via ``apply_to_state``.

        Returns
        -------
        jnp.ndarray
            Statevector of shape ``(2**n_qubits,)``.
        """
        dim = 2**n_qubits
        state = jnp.zeros(dim, dtype=jnp.complex128).at[0].set(1.0)
        for op in tape:
            state = op.apply_to_state(state, n_qubits)
        return state

    @staticmethod
    def _simulate_mixed(tape: List[Operation], n_qubits: int) -> jnp.ndarray:
        """
        Density-matrix simulation kernel.

        Starts from ρ = |00…0⟩⟨00…0| and applies each gate via
        ``apply_to_density`` (ρ → UρU† for unitaries, Σ_k K_k ρ K_k† for
        Kraus channels).  Required for noisy circuits.

        Returns
        -------
        jnp.ndarray
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
        """Apply the requested measurement to a pure statevector."""
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
        """Apply the requested measurement to a density matrix."""
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
        """
        Execute the circuit and return measurement results.

        Parameters
        ----------
        type : str
            ``"expval"``  – expectation value ⟨ψ|O|ψ⟩ / Tr(Oρ) for each
                            observable.
            ``"probs"``   – probability vector.
            ``"state"``   – raw statevector ``(2**n,)``.
            ``"density"`` – full density matrix ``(2**n, 2**n)``.
        obs : list[Operation]
            Observables (required for ``"expval"``).
        args : tuple
            Positional arguments forwarded to the circuit function *f*.
        kwargs : dict
            Keyword arguments forwarded to *f*.
        in_axes : tuple or None
            Batch axes for each element of *args*, following the same
            convention as ``jax.vmap``:

            * An integer selects that axis of the corresponding array as
              the batch dimension.
            * ``None`` means the argument is broadcast (not batched).

            When *in_axes* is provided, ``execute`` calls ``jax.vmap``
            over the pure simulation kernel and returns results with a
            leading batch dimension.

            Example — batch over axis 0 of a parameter array::

                script.execute(
                    type="expval",
                    obs=[PauliZ(0)],
                    args=(thetas,),
                    in_axes=(0,),
                )

            Example — batch over axis 2 of params, axis 0 of inputs
            (after ``_assimilate_batch`` has broadcast them to the same B)::

                script.execute(
                    type="expval",
                    obs=[PauliZ(0)],
                    args=(params, inputs),
                    in_axes=(2, 0),
                )

        Returns
        -------
        jnp.ndarray
            Without *in_axes*: shape depends on *type*.
            With *in_axes*: shape ``(B, ...)`` with a leading batch dim.

        Notes
        -----
        **Tape / kernel split** — the circuit function is run in Python
        *once* to record the tape and determine ``n_qubits`` and whether
        noise is present.  The pure JAX kernels (``_simulate_pure`` /
        ``_simulate_mixed``) are then vmapped.  This means Python
        overhead is O(circuit_depth), not O(B × circuit_depth).

        **shard_map migration** — the ``jax.vmap`` call in
        ``_execute_batched`` is the exact boundary to replace with
        ``jax.experimental.shard_map`` for multi-device execution::

            # Future drop-in replacement:
            from jax.experimental.shard_map import shard_map
            shard_map(_single_execute, mesh=mesh,
                      in_specs=..., out_specs=...)
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
        """
        Vectorise ``execute`` over a batch axis using ``jax.vmap``.

        The circuit function is traced **once** in Python with scalar
        slices to record the tape, determine ``n_qubits``, and detect
        noise.  The resulting pure simulation kernel is then vmapped over
        the requested axes.

        Parameters
        ----------
        in_axes : tuple
            One entry per element of *args*.  Follows ``jax.vmap``
            convention: an int gives the batch axis; ``None`` broadcasts.
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
        #
        # NOTE: this is the shard_map boundary.  To distribute across
        # multiple JAX devices in the future, replace the two lines below
        # with:
        #
        #   from jax.experimental.shard_map import shard_map
        #   from jax.sharding import PartitionSpec as P, Mesh
        #   result = shard_map(
        #       _single_execute, mesh=mesh,
        #       in_specs=tuple(P(0) if ax is not None else P() for ax in in_axes),
        #       out_specs=P(0),
        #   )(*args)
        #
        return jax.vmap(_single_execute, in_axes=in_axes)(*args)
