from typing import Optional, Any, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from qml_essentials import yaqsi as ys
from qml_essentials import operations as op
from qml_essentials.utils import logm_v
from qml_essentials.model import Model
import logging

log = logging.getLogger(__name__)


class Entanglement:
    @staticmethod
    def meyer_wallach(
        model: Model,
        n_samples: Optional[int | None],
        seed: Optional[int],
        scale: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        Calculates the entangling capacity of a given quantum circuit
        using Meyer-Wallach measure.

        Args:
            model (Model): The quantum circuit model.
            n_samples (Optional[int]): Number of samples per qubit.
                If None or < 0, the current parameters of the model are used.
            seed (Optional[int]): Seed for the random number generator.
            scale (bool): Whether to scale the number of samples.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Entangling capacity of the given circuit, guaranteed
                to be between 0.0 and 1.0.
        """
        if "noise_params" in kwargs:
            log.warning(
                "Meyer-Wallach measure not suitable for noisy circuits.\
                    Consider 'relative_entropy' instead."
            )

        if scale:
            n_samples = jnp.power(2, model.n_qubits) * n_samples

        random_key = jax.random.key(seed)
        if n_samples is not None and n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            random_key = model.initialize_params(random_key, repeat=n_samples)
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")

        # implicitly set input to none in case it's not needed
        kwargs.setdefault("inputs", None)
        # explicitly set execution type because everything else won't work
        rhos = model(execution_type="density", **kwargs).reshape(
            -1, 2**model.n_qubits, 2**model.n_qubits
        )

        ent = Entanglement._compute_meyer_wallach_meas(
            rhos, model.n_qubits, model.use_multithreading
        )

        log.debug(f"Variance of measure: {ent.var()}")

        return ent.mean()

    @staticmethod
    def _compute_meyer_wallach_meas(
        rhos: jnp.ndarray, n_qubits: int, use_multithreading: bool = False
    ) -> jnp.ndarray:
        """
        Computes the Meyer-Wallach entangling capability measure for a given
        set of density matrices.

        Args:
            rhos (jnp.ndarray): Density matrices of the sample quantum states.
                The shape is (B_s, 2^n, 2^n), where B_s is the number of samples
                (batch) and n the number of qubits
            n_qubits (int): The number of qubits
            use_multithreading (bool): Whether to use JAX vectorisation.

        Returns:
            jnp.ndarray: Entangling capability for each sample, array with
                shape (B_s,)
        """
        qb = list(range(n_qubits))

        def _f(rhos):
            entropy = 0
            for j in range(n_qubits):
                # Formula 6 in https://doi.org/10.48550/arXiv.quant-ph/0305094
                # Trace out qubit j, keep all others
                keep = qb[:j] + qb[j + 1 :]
                density = ys.partial_trace(rhos, n_qubits, keep)
                # only real values, because imaginary part will be separate
                # in all following calculations anyway
                # entropy should be 1/2 <= entropy <= 1
                entropy += jnp.trace((density @ density).real, axis1=-2, axis2=-1)

            # inverse averaged entropy and scale to [0, 1]
            return 2 * (1 - entropy / n_qubits)

        if use_multithreading:
            return jax.vmap(_f)(rhos)
        else:
            return _f(rhos)

    @staticmethod
    def bell_measurements(
        model: Model, n_samples: int, seed: int, scale: bool = False, **kwargs: Any
    ) -> float:
        """
        Compute the Bell measurement for a given model.

        Constructs a ``2 * n_qubits`` circuit that prepares two copies of
        the model state (on disjoint qubit registers), applies CNOTs and
        Hadamards, and measures probabilities on the first register.

        Args:
            model (Model): The quantum circuit model.
            n_samples (int): The number of samples to compute the measure for.
            seed (int): The seed for the random number generator.
            scale (bool): Whether to scale the number of samples
                according to the number of qubits.
            **kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: The Bell measurement value.
        """
        if "noise_params" in kwargs:
            log.warning(
                "Bell Measurements not suitable for noisy circuits. "
                "Consider 'relative_entropy' instead."
            )

        if scale:
            n_samples = jnp.power(2, model.n_qubits) * n_samples

        n = model.n_qubits

        def _bell_circuit(params, inputs, pulse_params=None, random_key=None, **kw):
            """Bell measurement circuit on 2*n qubits."""
            # First copy on wires 0..n-1
            model._variational(
                params, inputs, pulse_params=pulse_params, random_key=random_key, **kw
            )

            # Second copy on wires n..2n-1: record the tape then shift wires
            from qml_essentials.tape import recording as _recording

            with _recording() as shifted_tape:
                model._variational(
                    params,
                    inputs,
                    pulse_params=pulse_params,
                    random_key=random_key,
                    **kw,
                )
            for o in shifted_tape:
                shifted_op = o.__class__.__new__(o.__class__)
                shifted_op.__dict__.update(o.__dict__)
                shifted_op._wires = [w + n for w in o.wires]
                # Re-register on the active tape
                from qml_essentials.tape import active_tape as _active_tape

                tape = _active_tape()
                if tape is not None:
                    tape.append(shifted_op)

            # Bell measurement: CNOT + H
            for q in range(n):
                op.CX(wires=[q, q + n])
                op.H(wires=q)

        bell_script = ys.QuantumScript(f=_bell_circuit, n_qubits=2 * n)

        random_key = jax.random.key(seed)
        if n_samples is not None and n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            random_key = model.initialize_params(random_key, repeat=n_samples)
            params = model.params
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")

            if len(model.params.shape) <= 2:
                params = model.params.reshape(*model.params.shape, 1)
            else:
                log.info(f"Using sample size of model params: {model.params.shape[-1]}")
                params = model.params

        n_samples = params.shape[-1]
        inputs = model._inputs_validation(kwargs.get("inputs", None))

        # Execute: vmap over batch dimension of params (axis 2)
        if n_samples > 1:
            from qml_essentials.utils import safe_random_split

            random_keys = safe_random_split(random_key, num=n_samples)
            result = bell_script.execute(
                type="probs",
                args=(params, inputs, model.pulse_params, random_keys),
                in_axes=(2, None, None, 0),
            )
        else:
            result = bell_script.execute(
                type="probs",
                args=(params, inputs, model.pulse_params, random_key),
            )

        # Marginalize: for each qubit q, keep wires [q, q+n] from the 2n-qubit probs
        # The last probability in each pair gives P(|11⟩) for that qubit pair
        per_qubit = []
        for q in range(n):
            marg = ys.marginalize_probs(result, 2 * n, [q, q + n])
            per_qubit.append(marg)
        # per_qubit[q] has shape (n_samples, 4) or (4,)
        exp = jnp.stack(per_qubit, axis=-2)  # (..., n, 4)
        exp = 1 - 2 * exp[..., -1]  # (..., n)

        if not jnp.isclose(jnp.sum(exp.imag), 0, atol=1e-6):
            log.warning("Imaginary part of probabilities detected")
            exp = jnp.abs(exp)

        measure = 2 * (1 - exp.mean(axis=0))
        entangling_capability = min(max(float(measure.mean()), 0.0), 1.0)
        log.debug(f"Variance of measure: {measure.var()}")

        return entangling_capability

    @staticmethod
    def relative_entropy(
        model: Model,
        n_samples: int,
        n_sigmas: int,
        seed: Optional[int],
        scale: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        Calculates the relative entropy of entanglement of a given quantum
        circuit. This measure is also applicable to mixed state, albeit it
        might me not fully accurate in this simplified case.

        As the relative entropy is generally defined as the smallest relative
        entropy from the state in question to the set of separable states.
        However, as computing the nearest separable state is NP-hard, we select
        n_sigmas of random separable states to compute the distance to, which
        is not necessarily the nearest. Thus, this measure of entanglement
        presents an upper limit of entanglement.

        As the relative entropy is not necessarily between zero and one, this
        function also normalises by the relative entroy to the GHZ state.

        Args:
            model (Model): The quantum circuit model.
            n_samples (int): Number of samples per qubit.
                If <= 0, the current parameters of the model are used.
            n_sigmas (int): Number of random separable pure states to compare against.
            seed (Optional[int]): Seed for the random number generator.
            scale (bool): Whether to scale the number of samples.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Entangling capacity of the given circuit, guaranteed
                to be between 0.0 and 1.0.
        """
        dim = jnp.power(2, model.n_qubits)
        if scale:
            n_samples = dim * n_samples
            n_sigmas = dim * n_sigmas

        random_key = jax.random.key(seed)

        # Random separable states
        log_sigmas = sample_random_separable_states(
            model.n_qubits, n_samples=n_sigmas, random_key=random_key, take_log=True
        )

        random_key, _ = jax.random.split(random_key)

        if n_samples is not None and n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            model.initialize_params(random_key, repeat=n_samples)
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")

            if len(model.params.shape) <= 2:
                model.params = model.params.reshape(*model.params.shape, 1)
            else:
                log.info(f"Using sample size of model params: {model.params.shape[-1]}")

        rhos, log_rhos = Entanglement._compute_log_density(model, **kwargs)

        rel_entropies = jnp.zeros((n_sigmas, model.params.shape[-1]))

        for i, log_sigma in enumerate(log_sigmas):
            rel_entropies = rel_entropies.at[i].set(
                Entanglement._compute_rel_entropies(
                    rhos, log_rhos, log_sigma, model.use_multithreading
                )
            )

        # Entropy of GHZ states should be maximal
        ghz_model = Model(model.n_qubits, 1, "GHZ", data_reupload=False)
        rho_ghz, log_rho_ghz = Entanglement._compute_log_density(ghz_model, **kwargs)
        ghz_entropies = Entanglement._compute_rel_entropies(
            rho_ghz, log_rho_ghz, log_sigmas, use_multithreading=False
        )

        normalised_entropies = rel_entropies / ghz_entropies

        # Average all iterated states
        entangling_capability = normalised_entropies.T.min(axis=1)
        log.debug(f"Variance of measure: {entangling_capability.var()}")

        return entangling_capability.mean()

    @staticmethod
    def _compute_log_density(model: Model, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtains the density matrix of a model and computes its logarithm.

        Args:
            model (Model): The model for which to compute the density matrix.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - jnp.ndarray: density matrix.
                - jnp.ndarray: logarithm of the density matrix.
        """
        # implicitly set input to none in case it's not needed
        kwargs.setdefault("inputs", None)
        # explicitly set execution type because everything else won't work
        rho = model(execution_type="density", **kwargs)
        rho = rho.reshape(-1, 2**model.n_qubits, 2**model.n_qubits)
        log_rho = logm_v(rho) / jnp.log(2)
        return rho, log_rho

    @staticmethod
    def _compute_rel_entropies(
        rhos: jnp.ndarray,
        log_rhos: jnp.ndarray,
        log_sigmas: jnp.ndarray,
        use_multithreading: bool,
    ) -> jnp.ndarray:
        """
        Compute the relative entropy for a given model.

        Args:
            rhos (jnp.ndarray): Density matrix result of the circuit, has shape
                (R, 2^n, 2^n), with the batch size R and number of qubits n
            log_rhos (jnp.ndarray): Corresponding logarithm of the density
                matrix, has shape (R, 2^n, 2^n).
            log_sigmas (jnp.ndarray): Density matrix of next separable state,
                has shape (2^n, 2^n) if it's a single sigma or (S, 2^n, 2^n),
                with the batch size S (number of sigmas).

        Returns:
            jnp.ndarray: Relative Entropy for each sample
        """
        n_rhos = rhos.shape[0]
        if len(log_sigmas.shape) == 3:
            n_sigmas = log_sigmas.shape[0]
            rhos = jnp.tile(rhos, (n_sigmas, 1, 1))
            log_rhos = jnp.tile(log_rhos, (n_sigmas, 1, 1))
            einsum_subscript = "ij,jk->ik" if use_multithreading else "sij,sjk->sik"
        else:
            n_sigmas = 1
            log_sigmas = log_sigmas[jnp.newaxis, ...].repeat(n_rhos, axis=0)

        einsum_subscript = "ij,jk->ik" if use_multithreading else "sij,sjk->sik"

        def _f(rhos, log_rhos, log_sigmas):
            prod = jnp.einsum(einsum_subscript, rhos, log_rhos - log_sigmas)
            rel_entropies = jnp.abs(jnp.trace(prod, axis1=-2, axis2=-1))
            return rel_entropies

        if use_multithreading:
            rel_entropies = jax.vmap(_f, in_axes=(0, 0, 0))(rhos, log_rhos, log_sigmas)
        else:
            rel_entropies = _f(rhos, log_rhos, log_sigmas)

        if n_sigmas > 1:
            rel_entropies = rel_entropies.reshape(n_sigmas, n_rhos)
        return rel_entropies

    @staticmethod
    def entanglement_of_formation(
        model: Model,
        n_samples: int,
        seed: Optional[int],
        scale: bool = False,
        always_decompose: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        This function implements the entanglement of formation for mixed
        quantum systems.
        In that a mixed state gets decomposed into pure states with respective
        probabilities using the eigendecomposition of the density matrix.
        Then, the Meyer-Wallach measure is computed for each pure state,
        weighted by the eigenvalue.
        See e.g. https://doi.org/10.48550/arXiv.quant-ph/0504163

        Note that the decomposition is *not unique*! Therefore, this measure
        presents the entanglement for *some* decomposition into pure states,
        not necessarily the one that is anticipated when applying the Kraus
        channels.
        If a pure state is provided, this results in the same value as the
        Entanglement.meyer_wallach function if `always_decompose` flag is not set.

        Args:
            model (Model): The quantum circuit model.
            n_samples (int): Number of samples per qubit.
            seed (Optional[int]): Seed for the random number generator.
            scale (bool): Whether to scale the number of samples.
            always_decompose (bool): Whether to explicitly compute the
                entantlement of formation for the eigendecomposition of a pure
                state.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Entangling capacity of the given circuit, guaranteed
                to be between 0.0 and 1.0.
        """

        if scale:
            n_samples = jnp.power(2, model.n_qubits) * n_samples

        random_key = jax.random.key(seed)
        if n_samples is not None and n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            model.initialize_params(random_key, repeat=n_samples)
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")

            if len(model.params.shape) <= 2:
                model.params = model.params.reshape(*model.params.shape, 1)
            else:
                log.info(f"Using sample size of model params: {model.params.shape[-1]}")

        # implicitly set input to none in case it's not needed
        kwargs.setdefault("inputs", None)
        rhos = model(execution_type="density", **kwargs)
        rhos = rhos.reshape(-1, 2**model.n_qubits, 2**model.n_qubits)
        ent = Entanglement._compute_entanglement_of_formation(
            rhos, model.n_qubits, always_decompose, model.use_multithreading
        )
        return ent.mean()

    @staticmethod
    def _compute_entanglement_of_formation(
        rhos: jnp.ndarray,
        n_qubits: int,
        always_decompose: bool,
        use_multithreading: bool,
    ) -> jnp.ndarray:
        """
        Computes the entanglement of formation for a given batch of density
        matrices.

        Args:
            rho (jnp.ndarray): The density matrices, has shape (B_s, 2^n, 2^n),
                where B_s is the batch size and n the number of qubits.
            n_qubits (int): Number of qubits
            always_decompose (bool): Whether to explicitly compute the
                entantlement of formation for the eigendecomposition of a pure
                state.
            use_multithreading (bool): Whether to use JAX vectorisation.

        Returns:
            jnp.ndarray: Entanglement for the provided density matrices.
        """
        eigenvalues, eigenvectors = jnp.linalg.eigh(rhos)
        if not always_decompose and jnp.isclose(eigenvalues, 1.0).any(axis=-1).all():
            return Entanglement._compute_meyer_wallach_meas(
                rhos, n_qubits, use_multithreading
            )

        rhos = np.einsum("sij,sik->sijk", eigenvectors, eigenvectors.conjugate())
        measures = Entanglement._compute_meyer_wallach_meas(
            rhos.reshape(-1, 2**n_qubits, 2**n_qubits), n_qubits, use_multithreading
        )
        ent = np.einsum("si,si->s", measures.reshape(-1, 2**n_qubits), eigenvalues)
        return ent

    @staticmethod
    def concentratable_entanglement(
        model: Model, n_samples: int, seed: int, scale: bool = False, **kwargs: Any
    ) -> float:
        """
        Computes the concentratable entanglement of a given model.

        This method utilizes the Concentratable Entanglement measure from
        https://arxiv.org/abs/2104.06923.  The swap test is implemented
        directly in yaqsi using a ``3 * n_qubits`` circuit.

        Args:
            model (Model): The quantum circuit model.
            n_samples (int): The number of samples to compute the measure for.
            seed (int): The seed for the random number generator.
            scale (bool): Whether to scale the number of samples according to
                the number of qubits.
            **kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Entangling capability of the given circuit, guaranteed
                to be between 0.0 and 1.0.
        """
        n = model.n_qubits
        N = 2**n

        if scale:
            n_samples = N * n_samples

        def _shift_and_append(tape_ops, offset):
            """Re-register *tape_ops* on the active tape with wires shifted."""
            from qml_essentials.tape import active_tape as _active_tape

            current = _active_tape()
            if current is None:
                return
            for o in tape_ops:
                shifted = o.__class__.__new__(o.__class__)
                shifted.__dict__.update(o.__dict__)
                shifted._wires = [w + offset for w in o.wires]
                current.append(shifted)

        def _swap_test_circuit(
            params, inputs, pulse_params=None, random_key=None, **kw
        ):
            """Swap-test circuit on 3*n qubits."""
            from qml_essentials.tape import recording as _recording

            # First copy on wires n..2n-1
            with _recording() as copy1_tape:
                model._variational(
                    params,
                    inputs,
                    pulse_params=pulse_params,
                    random_key=random_key,
                    **kw,
                )
            _shift_and_append(copy1_tape, n)

            # Second copy on wires 2n..3n-1
            with _recording() as copy2_tape:
                model._variational(
                    params,
                    inputs,
                    pulse_params=pulse_params,
                    random_key=random_key,
                    **kw,
                )
            _shift_and_append(copy2_tape, 2 * n)

            # Swap test: H on ancilla register (wires 0..n-1)
            for i in range(n):
                op.H(wires=i)

            for i in range(n):
                op.CSWAP(wires=[i, i + n, i + 2 * n])

            for i in range(n):
                op.H(wires=i)

        swap_script = ys.QuantumScript(f=_swap_test_circuit, n_qubits=3 * n)

        random_key = jax.random.key(seed)
        if n_samples is not None and n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            model.initialize_params(random_key, repeat=n_samples)
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")

            if len(model.params.shape) <= 2:
                model.params = model.params.reshape(*model.params.shape, 1)
            else:
                log.info(f"Using sample size of model params: {model.params.shape[-1]}")

        params = model.params
        inputs = model._inputs_validation(kwargs.get("inputs", None))
        n_batch = params.shape[-1]

        if n_batch > 1:
            from qml_essentials.utils import safe_random_split

            random_keys = safe_random_split(random_key, num=n_batch)
            probs = swap_script.execute(
                type="probs",
                args=(params, inputs, model.pulse_params, random_keys),
                in_axes=(2, None, None, 0),
            )
        else:
            probs = swap_script.execute(
                type="probs",
                args=(params, inputs, model.pulse_params, random_key),
            )

        # Marginalize to the ancilla register (wires 0..n-1)
        probs = ys.marginalize_probs(probs, 3 * n, list(range(n)))

        ent = 1 - probs[..., 0]

        log.debug(f"Variance of measure: {ent.var()}")

        return float(ent.mean())


def sample_random_separable_states(
    n_qubits: int,
    n_samples: int,
    random_key: jax.random.PRNGKey,
    take_log: bool = False,
) -> jnp.ndarray:
    """
    Sample random separable states (density matrix).

    Args:
        n_qubits (int): number of qubits in the state
        n_samples (int): number of states
        random_key (random.PRNGKey): JAX random key
        take_log (bool): if the matrix logarithm of the density matrix should be taken.

    Returns:
        jnp.ndarray: Density matrices of shape (n_samples, 2**n_qubits, 2**n_qubits)
    """
    model = Model(n_qubits, 1, "No_Entangling", data_reupload=False)
    model.initialize_params(random_key, repeat=n_samples)
    # explicitly set execution type because everything else won't work
    sigmas = model(execution_type="density", inputs=None)
    if take_log:
        sigmas = logm_v(sigmas) / jnp.log(2.0 + 0j)

    return sigmas
