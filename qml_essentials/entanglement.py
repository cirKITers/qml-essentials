from typing import Optional, Any, List
import pennylane as qml
import pennylane.numpy as np
from copy import deepcopy
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
            n_samples = np.power(2, model.n_qubits) * n_samples

        rng = np.random.default_rng(seed)
        if n_samples is not None and n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            # TODO: maybe switch to JAX rng
            model.initialize_params(rng=rng, repeat=n_samples)
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")

        # implicitly set input to none in case it's not needed
        kwargs.setdefault("inputs", None)
        # explicitly set execution type because everything else won't work
        rhos = model(execution_type="density", **kwargs).reshape(
            -1, 2**model.n_qubits, 2**model.n_qubits
        )

        mw_measure = np.zeros(len(rhos))

        for i, rho in enumerate(rhos):
            mw_measure[i] = Entanglement._compute_meyer_wallach_meas(
                rho, model.n_qubits
            )

        # Average all iterated states
        entangling_capability = min(max(mw_measure.mean(), 0.0), 1.0)
        log.debug(f"Variance of measure: {mw_measure.var()}")

        # catch floating point errors
        return float(entangling_capability)

    @staticmethod
    def _compute_meyer_wallach_meas(rho: np.ndarray, n_qubits: int):
        qb = list(range(n_qubits))
        entropy = 0
        for j in range(n_qubits):
            # Formula 6 in https://doi.org/10.48550/arXiv.quant-ph/0305094
            density = qml.math.partial_trace(rho, qb[:j] + qb[j + 1 :])
            # only real values, because imaginary part will be separate
            # in all following calculations anyway
            # entropy should be 1/2 <= entropy <= 1
            entropy += np.trace((density @ density).real)

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
            n_samples = np.power(2, model.n_qubits) * n_samples

        def _circuit(params: np.ndarray, inputs: np.ndarray) -> List[np.ndarray]:
            """
            Compute the Bell measurement circuit.

            Args:
                params (np.ndarray): The model parameters.
                inputs (np.ndarray): The input to the model.

            Returns:
                List[np.ndarray]: The probabilities of the Bell measurement.
            """
            model._variational(params, inputs)

            qml.map_wires(
                model._variational,
                {i: i + model.n_qubits for i in range(model.n_qubits)},
            )(params, inputs)

            for q in range(model.n_qubits):
                qml.CNOT(wires=[q, q + model.n_qubits])
                qml.H(q)

            obs_wires = [(q, q + model.n_qubits) for q in range(model.n_qubits)]
            return [qml.probs(wires=w) for w in obs_wires]

        model.circuit = qml.QNode(
            _circuit,
            qml.device(
                "default.qubit",
                shots=model.shots,
                wires=model.n_qubits * 2,
            ),
        )

        rng = np.random.default_rng(seed)
        if n_samples is not None and n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            # TODO: maybe switch to JAX rng
            model.initialize_params(rng=rng, repeat=n_samples)
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
        mw_measure = np.zeros(n_samples)

        for i in range(n_samples):
            # implicitly set input to none in case it's not needed
            kwargs.setdefault("inputs", None)
            exp = model(params=params[:, :, i], **kwargs)

            exp = 1 - 2 * exp[:, -1]
            mw_measure[i] = 2 * (1 - exp.mean())
        entangling_capability = min(max(mw_measure.mean(), 0.0), 1.0)
        log.debug(f"Variance of measure: {mw_measure.var()}")

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
        dim = np.power(2, model.n_qubits)
        if scale:
            n_samples = dim * n_samples
            n_sigmas = dim * n_sigmas

        rng = np.random.default_rng(seed)

        # Random separable states
        log_sigmas = sample_random_separable_states(
            model.n_qubits, n_samples=n_sigmas, rng=rng, take_log=True
        )

        if n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            model.initialize_params(rng=rng, repeat=n_samples)
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")

            if len(model.params.shape) <= 2:
                model.params = model.params.reshape(*model.params.shape, 1)
            else:
                log.info(f"Using sample size of model params: {model.params.shape[-1]}")

        ghz_model = Model(model.n_qubits, 1, "GHZ", data_reupload=False)

        normalised_entropies = np.zeros((n_sigmas, n_samples))
        for j, log_sigma in enumerate(log_sigmas):

            # Entropy of GHZ states should be maximal
            ghz_entropy = Entanglement._compute_rel_entropies(
                ghz_model,
                log_sigma,
            )

            rel_entropy = Entanglement._compute_rel_entropies(
                model, log_sigma, **kwargs
            )

            normalised_entropies[j] = rel_entropy / ghz_entropy

        # Average all iterated states
        entangling_capability = normalised_entropies.min(axis=0).mean()
        log.debug(f"Variance of measure: {normalised_entropies.var()}")

        return entangling_capability

    @staticmethod
    def _compute_rel_entropies(
        model: Model,
        log_sigma: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute the relative entropy for a given model.

        Args:
            model (Model): The model for which to compute entanglement
            log_sigma (np.ndarray): Density matrix of next separable state

        Returns:
            np.ndarray: Relative Entropy for each sample
        """
        # implicitly set input to none in case it's not needed
        kwargs.setdefault("inputs", None)
        # explicitly set execution type because everything else won't work
        rho = model(execution_type="density", **kwargs)
        rho = rho.reshape(-1, 2**model.n_qubits, 2**model.n_qubits)
        log_rho = logm_v(rho) / np.log(2)

        rel_entropies = np.abs(np.trace(rho @ (log_rho - log_sigma), axis1=1, axis2=2))

        return rel_entropies

    @staticmethod
    def entanglement_of_formation(
        model: Model,
        n_samples: int,
        seed: Optional[int],
        scale: bool = False,
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
        Entanglement.meyer_wallach function.

        Args:
            model (Model): The quantum circuit model.
            n_samples (int): Number of samples per qubit.
            seed (Optional[int]): Seed for the random number generator.
            scale (bool): Whether to scale the number of samples.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Entangling capacity of the given circuit, guaranteed
                to be between 0.0 and 1.0.
        """

        if scale:
            n_samples = np.power(2, model.n_qubits) * n_samples

        rng = np.random.default_rng(seed)
        if n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            model.initialize_params(rng=rng, repeat=n_samples)
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
        entanglement = np.zeros(len(rhos))
        for i, rho in enumerate(rhos):
            entanglement[i] = Entanglement._compute_entanglement_of_formation(
                rho, model.n_qubits
            )
        entangling_capability = min(max(entanglement.mean(), 0.0), 1.0)
        return float(entangling_capability)

    @staticmethod
    def _compute_entanglement_of_formation(rho: np.ndarray, n_qubits: int) -> float:
        """
        Computes the entanglement of formation for a given density matrix rho.

        Args:
            rho (np.ndarray): The density matrix
            n_qubits (int): Number of qubits

        Returns:
            float: Entanglement for the provided state.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        if any(np.isclose(eigenvalues, 1.0)):  # Pure state
            return Entanglement._compute_meyer_wallach_meas(rho, n_qubits)
        ent = 0
        for prob, ev in zip(eigenvalues, eigenvectors):
            ev = ev.reshape(-1, 1)
            rho = ev @ np.conjugate(ev).T
            mw_measure = Entanglement._compute_meyer_wallach_meas(rho, n_qubits)
            ent += prob * mw_measure
        return ent


def sample_random_separable_states(
    n_qubits: int, n_samples: int, rng: np.random.Generator, take_log: bool = False
) -> np.ndarray:
    """
    Sample random separable states (density matrix).

    Args:
        n_qubits (int): number of qubits in the state
        n_samples (int): number of states
        rng (np.random.Generator): random number generator
        take_log (bool): if the matrix logarithm of the density matrix should be taken.

    Returns:
        np.ndarray: Density matrices of shape (n_samples, 2**n_qubits, 2**n_qubits)
    """
    model = Model(n_qubits, 1, "No_Entangling", data_reupload=False)
    model.initialize_params(rng=rng, repeat=n_samples)
    # explicitly set execution type because everything else won't work
    sigmas = model(execution_type="density", inputs=None)
    if take_log:
        sigmas = logm_v(sigmas) / np.log(2)

    return sigmas
