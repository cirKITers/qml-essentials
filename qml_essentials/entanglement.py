from typing import Optional, Any, List, Tuple
import pennylane as qml
import jax.numpy as jnp
import numpy as np
from jax import random

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

        random_key = random.key(seed)
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

        ent = Entanglement._compute_meyer_wallach_meas(rhos, model.n_qubits)

        log.debug(f"Variance of measure: {ent.var()}")

        return ent.mean()

    @staticmethod
    def _compute_meyer_wallach_meas(rhos: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """
        Computes the Meyer-Wallach entangling capability measure for a given
        set of density matrices.

        Args:
            rhos (jnp.ndarray): Density matrices of the sample quantum states.
                The shape is (B_s, 2^n, 2^n), where B_s is the number of samples
                (batch) and n the number of qubits
            n_qubits (int): The number of qubits

        Returns:
            jnp.ndarray: Entangling capability for each sample, array with
                shape (B_s,)
        """
        qb = list(range(n_qubits))
        entropy = 0
        for j in range(n_qubits):
            # Formula 6 in https://doi.org/10.48550/arXiv.quant-ph/0305094
            density = qml.math.partial_trace(rhos, qb[:j] + qb[j + 1 :])
            # only real values, because imaginary part will be separate
            # in all following calculations anyway
            # entropy should be 1/2 <= entropy <= 1
            entropy += jnp.trace((density @ density).real, axis1=1, axis2=2)

        # inverse averaged entropy and scale to [0, 1]
        return 2 * (1 - entropy / n_qubits)

    @staticmethod
    def bell_measurements(
        model: Model, n_samples: int, seed: int, scale: bool = False, **kwargs: Any
    ) -> float:
        """
        Compute the Bell measurement for a given model.

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
                "Bell Measurements not suitable for noisy circuits.\
                    Consider 'relative_entropy' instead."
            )

        if scale:
            n_samples = jnp.power(2, model.n_qubits) * n_samples

        def _circuit(
            params: jnp.ndarray, inputs: jnp.ndarray, **kwargs
        ) -> List[jnp.ndarray]:
            """
            Compute the Bell measurement circuit.

            Args:
                params (jnp.ndarray): The model parameters.
                inputs (jnp.ndarray): The input to the model.
                pulse_params (jnp.ndarray): The model pulse parameters.
                enc_params (Optional[jnp.ndarray]): The frequency encoding parameters.

            Returns:
                List[jnp.ndarray]: The probabilities of the Bell measurement.
            """
            model._variational(params, inputs, **kwargs)

            qml.map_wires(
                model._variational,
                {i: i + model.n_qubits for i in range(model.n_qubits)},
            )(params, inputs)

            for q in range(model.n_qubits):
                qml.CNOT(wires=[q, q + model.n_qubits])
                qml.H(q)

            # look at the auxiliary qubits
            return model._observable()

        prev_output_qubit = model.output_qubit
        model.output_qubit = [(q, q + model.n_qubits) for q in range(model.n_qubits)]
        model.circuit = qml.QNode(
            _circuit,
            qml.device(
                "default.qubit",
                shots=model.shots,
                wires=model.n_qubits * 2,
            ),
        )

        random_key = random.key(seed)
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
        measure = jnp.zeros(n_samples)

        # implicitly set input to none in case it's not needed
        kwargs.setdefault("inputs", None)
        exp = model(params=params, execution_type="probs", **kwargs)
        exp = 1 - 2 * exp[..., -1]

        if not jnp.isclose(jnp.sum(exp.imag), 0, atol=1e-6):
            log.warning("Imaginary part of probabilities detected")
            exp = jnp.abs(exp)

        measure = 2 * (1 - exp.mean(axis=0))
        entangling_capability = min(max(measure.mean(), 0.0), 1.0)
        log.debug(f"Variance of measure: {measure.var()}")

        # restore state
        model.output_qubit = prev_output_qubit
        return float(entangling_capability)

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

        random_key = random.key(seed)

        # Random separable states
        log_sigmas = sample_random_separable_states(
            model.n_qubits, n_samples=n_sigmas, random_key=random_key, take_log=True
        )

        random_key, _ = random.split(random_key)

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

        normalised_entropies = jnp.zeros((n_sigmas, model.params.shape[-1]))

        rel_entropies = Entanglement._compute_rel_entropies(rhos, log_rhos, log_sigmas)

        # Entropy of GHZ states should be maximal
        ghz_model = Model(model.n_qubits, 1, "GHZ", data_reupload=False)
        rho_ghz, log_rho_ghz = Entanglement._compute_log_density(ghz_model, **kwargs)
        ghz_entropies = Entanglement._compute_rel_entropies(
            rho_ghz, log_rho_ghz, log_sigmas
        )
        ghz_min_dist = jnp.min(ghz_entropies)

        normalised_entropies = rel_entropies / ghz_min_dist

        # Average all iterated states
        entangling_capability = normalised_entropies.min(axis=1).mean()
        log.debug(f"Variance of measure: {normalised_entropies.var()}")

        return entangling_capability

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
    ) -> jnp.ndarray:
        """
        Compute the relative entropy for a given model.

        Args:
            rhos (jnp.ndarray): Density matrix result of the circuit, has shape
                (R, 2^n, 2^n), with the batch size R and number of qubits n
            log_rhos (jnp.ndarray): Corresponding logarithm of the density
                matrix, has shape (R, 2^n, 2^n).
            log_sigmas (jnp.ndarray): Density matrix of next separable state,
                has shape (S, 2^n, 2^n), with the batch size S (number of
                sigmas).

        Returns:
            jnp.ndarray: Relative Entropy for each sample
        """
        n_rhos = rhos.shape[0]
        n_sigmas = log_sigmas.shape[0]

        rhos = jnp.tile(rhos, (n_sigmas, 1, 1))
        log_rhos = jnp.tile(log_rhos, (n_sigmas, 1, 1))
        log_sigmas = log_sigmas.repeat(n_rhos, axis=0)
        prod = jnp.einsum("sij,sjk->sik", rhos, log_rhos - log_sigmas)
        rel_entropies = jnp.abs(jnp.trace(prod, axis1=1, axis2=2))
        rel_entropies = rel_entropies.reshape(n_sigmas, n_rhos).T

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

        random_key = random.key(seed)
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
            rhos, model.n_qubits, always_decompose
        )
        return ent.mean()

    @staticmethod
    def _compute_entanglement_of_formation(
        rhos: jnp.ndarray, n_qubits: int, always_decompose: bool
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

        Returns:
            jnp.ndarray: Entanglement for the provided density matrices.
        """
        ent = jnp.zeros(len(rhos))
        eigenvalues, eigenvectors = jnp.linalg.eigh(rhos)
        if not always_decompose and jnp.isclose(eigenvalues, 1.0).any(axis=-1).all():
            return Entanglement._compute_meyer_wallach_meas(rhos, n_qubits)

        rhos = np.einsum("sij,sik->sijk", eigenvectors, eigenvectors.conjugate())
        measures = Entanglement._compute_meyer_wallach_meas(
            rhos.reshape(-1, 2**n_qubits, 2**n_qubits), n_qubits
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
        https://arxiv.org/abs/2104.06923.

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

        dev = qml.device(
            "default.mixed",
            shots=model.shots,
            wires=n * 3,
        )

        @qml.qnode(device=dev)
        def _swap_test(
            params: jnp.ndarray, inputs: jnp.ndarray, **kwargs
        ) -> jnp.ndarray:
            """
            Constructs a circuit to compute the concentratable entanglement using the
            swap test by creating two copies of a state given by a density matrix rho
            and mapping the output wires accordingly.

            Args:
                rho (jnp.ndarray): the density matrix of the state on which the swap
                    test is performed.

            Returns:
                List[jnp.ndarray]: Probabilities obtained from the swap test circuit.
            """

            qml.map_wires(model._variational, wire_map={o: o + n for o in range(n)})(
                params, inputs, **kwargs
            )
            qml.map_wires(
                model._variational, wire_map={o: o + 2 * n for o in range(n)}
            )(params, inputs, **kwargs)

            # Perform swap test
            for i in range(n):
                qml.H(i)

            for i in range(n):
                qml.CSWAP([i, i + n, i + 2 * n])

            for i in range(n):
                qml.H(i)

            return qml.probs(wires=[i for i in range(n)])

        random_key = random.key(seed)
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

        probs = _swap_test(model.params, model._inputs_validation(None), **kwargs)
        ent = 1 - probs[..., 0]

        # Catch floating point errors
        log.debug(f"Variance of measure: {ent.var()}")

        return ent.mean()


def sample_random_separable_states(
    n_qubits: int, n_samples: int, random_key: random.PRNGKey, take_log: bool = False
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
