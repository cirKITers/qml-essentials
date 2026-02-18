from __future__ import annotations

from typing import Callable, List, Optional, Union

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

    # -- core execution -------------------------------------------------
    def execute(
        self,
        type: str = "expval",
        obs: Optional[List[Operation]] = None,
        *,
        args: tuple = (),
        kwargs: dict = {},
    ) -> jnp.ndarray:
        """
        Execute the circuit and return measurement results.

        Parameters
        ----------
        type : str
            ``"expval"``  – expectation value ⟨ψ|O|ψ⟩ / Tr(Oρ) for each
                            observable.  Uses statevector simulation.
            ``"probs"``   – probability vector.  Uses statevector simulation.
            ``"state"``   – raw statevector ``(2**n,)``.
            ``"density"`` – full density matrix ``(2**n, 2**n)``.  Required
                            for noise simulation (future Kraus channels).
        obs : list[Operation]
            Observables (required for ``"expval"``).

        Returns
        -------
        jnp.ndarray
        """
        if obs is None:
            obs = []

        # 1. Record the operations
        tape = self._record(*args, **kwargs)

        # 2. Determine system size
        n_qubits = self._n_qubits or self._infer_n_qubits(tape, obs)
        dim = 2**n_qubits

        # 3. If the tape contains any Kraus channel (or user asked for density),
        #    we must use the density-matrix kernel.
        has_noise = any(isinstance(op, KrausChannel) for op in tape)
        if type == "density" or has_noise:
            rho = jnp.zeros((dim, dim), dtype=jnp.complex128).at[0, 0].set(1.0)
            for op in tape:
                rho = op.apply_to_density(rho, n_qubits)
            if type == "density":
                return rho
            if type == "probs":
                return jnp.real(jnp.diag(rho))
            if type == "expval":
                # Tr(O ρ) for each observable O
                # Reuse apply_to_state on each column of ρ via vmap to form O·ρ,
                # then take the trace.  For single-qubit observables this is fast.
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

        # All other measurement types use the statevector kernel
        state = jnp.zeros(dim, dtype=jnp.complex128).at[0].set(1.0)
        for op in tape:
            state = op.apply_to_state(state, n_qubits)

        if type == "state":
            return state

        if type == "probs":
            return jnp.abs(state) ** 2

        if type == "expval":
            # ⟨ψ|O|ψ⟩  =  Re(⟨ψ| (O|ψ⟩))  — no full matrix needed
            return jnp.array(
                [
                    jnp.real(jnp.vdot(state, ob.apply_to_state(state, n_qubits)))
                    for ob in obs
                ]
            )

        raise ValueError(f"Unknown measurement type: {type!r}")
