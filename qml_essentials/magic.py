from typing import Optional, Any
import string
import logging

import jax
import jax.numpy as jnp

from qml_essentials import operations as op
from qml_essentials.model import Model
from qml_essentials.pauli import PauliCircuit
from qml_essentials.tape import recording

log = logging.getLogger(__name__)


class Magic:
    r"""Magic (nonstabilizerness) of a quantum circuit.

    The primary measure is the second-order stabilizer Renyi entropy

    .. math::
        M_2(\lvert\psi\rangle) =
        -\log\!\Big(\frac{1}{2^n}\sum_{P\in\mathcal P_n}
        \langle\psi\lvert P\rvert\psi\rangle^{4}\Big),

    where the sum runs over all $4^n$ $n$-qubit Pauli strings $\mathcal P_n$.
    $M_2$ is non-negative and vanishes if and only if the state is a stabilizer
    state.  It is a nonstabilizerness monotone, requires no minimisation, and is
    differentiable, which makes it the de-facto standard magic measure for
    parameterised quantum circuits.
    """

    @classmethod
    def stabilizer_renyi_entropy(
        cls,
        model: Model,
        n_samples: Optional[int | None],
        random_key: Optional[jax.random.PRNGKey] = None,
        scale: bool = False,
        **kwargs: Any,
    ) -> float:
        r"""Second-order stabilizer Renyi entropy $M_2$ of a given model.

        $M_2$ is faithful only for pure states.  The states are obtained via
        ``execution_type="state"``; passing ``noise_params`` (mixed states)
        yields values that are not a valid magic measure.

        Args:
            model (Model): The quantum circuit model.
            n_samples (Optional[int]): Number of parameter samples to average
                over.  If None or < 0, the current parameters of the model are
                used.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for
                parameter initialization.  If None, uses the model's internal
                random key.
            scale (bool): Whether to scale the number of samples by $2^n$.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Mean $M_2$ over the sampled states, non-negative and zero
                for stabilizer states.
        """
        if "noise_params" in kwargs:
            log.warning(
                "Stabilizer Renyi entropy is only a faithful magic measure for "
                "pure states; results for noisy circuits are not meaningful."
            )

        if scale:
            n_samples = jnp.power(2, model.n_qubits) * n_samples

        if n_samples is not None and n_samples > 0:
            random_key = model.initialize_params(random_key, repeat=n_samples)

        # implicitly set input to none in case it's not needed
        kwargs.setdefault("inputs", None)
        # explicitly request statevectors; the measure is defined on pure states
        states = model(execution_type="state", **kwargs).reshape(-1, 2**model.n_qubits)

        n_qubits = model.n_qubits
        m2 = jax.vmap(lambda psi: cls._compute_m2(psi, n_qubits))(states)

        log.debug(f"Variance of measure: {m2.var()}")

        return m2.mean()

    @classmethod
    def _compute_m2(cls, psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        r"""Compute $M_2$ for a single pure statevector.

        Args:
            psi (jnp.ndarray): Statevector of length $2^n$.
            n_qubits (int): Number of qubits $n$.

        Returns:
            jnp.ndarray: Scalar $M_2$ value.
        """
        exp = cls._pauli_expectations(psi, n_qubits)
        d = 2**n_qubits
        return -jnp.log(jnp.sum(exp**4) / d)

    @staticmethod
    def _pauli_expectations(psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        r"""Return all $4^n$ Pauli expectation values $\langle\psi|P|\psi\rangle$.

        Evaluated as a vectorised Pauli transform that contracts each qubit's
        index pair against the four single-qubit Paulis, avoiding both the dense
        $2^n\times2^n$ operators and a Python loop over the $4^n$ strings.  The
        intermediate scales as $O(4^n)$, so the exact measure is intended for
        small-to-moderate qubit counts.  The result is real for Hermitian Pauli
        words and a pure state.

        Args:
            psi (jnp.ndarray): Statevector of length $2^n$.
            n_qubits (int): Number of qubits $n$.

        Returns:
            jnp.ndarray: Real array of length $4^n$ ordered as I, X, Y, Z per
                qubit (qubit 0 most significant).
        """
        paulis = jnp.stack(op._PAULI_MATS)  # (4, 2, 2), order I, X, Y, Z
        psi_t = psi.reshape((2,) * n_qubits)

        letters = string.ascii_letters
        if 3 * n_qubits > len(letters):
            raise ValueError(
                f"Exact stabilizer Renyi entropy supports at most "
                f"{len(letters) // 3} qubits."
            )
        out_idx = letters[:n_qubits]
        bra_idx = letters[n_qubits : 2 * n_qubits]
        ket_idx = letters[2 * n_qubits : 3 * n_qubits]

        operands = []
        subs = []
        # <psi|P|psi> = sum_{i,j} conj(psi)_i ( prod_k P^{(k)}_{i_k j_k} ) psi_j
        for k in range(n_qubits):
            operands.append(paulis)
            subs.append(out_idx[k] + bra_idx[k] + ket_idx[k])
        operands.append(jnp.conj(psi_t))
        subs.append(bra_idx)
        operands.append(psi_t)
        subs.append(ket_idx)

        einsum_str = ",".join(subs) + "->" + out_idx
        exp = jnp.einsum(einsum_str, *operands, optimize=True)
        return exp.reshape(-1).real

    @classmethod
    def non_clifford_count(
        cls, model: Model, inputs: Optional[jnp.ndarray] = None
    ) -> int:
        """Number of non-Clifford (Pauli-rotation) gates in the circuit.

        This is a coarse structural proxy for magic, not a faithful magic
        measure: it counts the Pauli-rotation gates after decomposing the tape
        into Clifford and Pauli-rotation gates, independent of their angles, so
        it is roughly constant across parameters.  Prefer
        ``stabilizer_renyi_entropy`` for an actual magic value.

        Args:
            model (Model): The quantum circuit model.
            inputs (Optional[jnp.ndarray]): Inputs used to build the circuit.
                If None, default zero inputs are used.

        Returns:
            int: Count of Pauli-rotation gates in the decomposed circuit.
        """
        inputs = model._inputs_validation(inputs)

        # Record the unitary tape only; noise channels are irrelevant here.
        saved_noise = model._noise_params
        model._noise_params = None
        try:
            with recording() as tape:
                model._variational(
                    model.params[0] if model.params.ndim == 3 else model.params,
                    inputs[0] if inputs.ndim == 2 else inputs,
                    noise_params=None,
                )
        finally:
            model._noise_params = saved_noise

        ops = PauliCircuit.get_clifford_pauli_gates(tape)
        return sum(isinstance(o, PauliCircuit.PAULI_ROTATION_GATES) for o in ops)
