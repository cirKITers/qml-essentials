import jax.numpy as jnp
import jax
import numpy as np
from typing import Tuple, Any, Optional
from scipy import integrate
from scipy.linalg import sqrtm
from scipy.special import rel_entr
from qml_essentials.model import Model
import os


class Expressibility:
    @classmethod
    def _sample_state_fidelities(
        cls,
        model: Model,
        n_samples: int,
        random_key: Optional[jax.random.PRNGKey] = None,
        kwargs: Any = None,
    ) -> jnp.ndarray:
        """
        Compute the fidelities for each parameter set.

        Args:
            model (Callable): Function that models the quantum circuit.
            n_samples (int): Number of parameter sets to generate.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for
                parameter initialization. If None, uses the model's internal
                random key.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            jnp.ndarray: Array of shape (n_samples,) containing the fidelities.
        """
        # Generate random parameter sets
        # We need two sets of parameters, as we are computing fidelities for a
        # pair of random state vectors
        model.initialize_params(random_key, repeat=n_samples * 2)

        # Evaluate the model for all parameters
        # Execution type is explicitly set to density
        sv: jnp.ndarray = model(
            params=model.params,
            execution_type="density",
            **kwargs,
        )

        # $\sqrt{\rho}$
        sqrt_sv1: jnp.ndarray = jnp.array([sqrtm(m) for m in sv[:n_samples]])

        # $\sqrt{\rho} \sigma \sqrt{\rho}$
        inner_fidelity = sqrt_sv1 @ sv[n_samples:] @ sqrt_sv1

        # Compute the fidelity using the partial trace of the statevector
        fidelity: jnp.ndarray = (
            jnp.trace(
                jnp.array([sqrtm(m) for m in inner_fidelity]),
                axis1=1,
                axis2=2,
            )
            ** 2
        )

        fidelity = jnp.abs(fidelity)

        return fidelity

    @classmethod
    def state_fidelities(
        cls,
        n_samples: int,
        n_bins: int,
        model: Model,
        random_key: Optional[jax.random.PRNGKey] = None,
        scale: bool = False,
        **kwargs: Any,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample the state fidelities and histogram them into a 2D array.

        Args:
            n_samples (int): Number of parameter sets to generate.
            n_bins (int): Number of histogram bins.
            model (Callable): Function that models the quantum circuit.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for
                parameter initialization. If None, uses the model's internal
                random key.
            scale (bool): Whether to scale the number of samples and bins.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Tuple containing the bin edges,
            and histogram values.
        """
        if scale:
            n_samples = jnp.power(2, model.n_qubits) * n_samples
            n_bins = model.n_qubits * n_bins

        fidelities = cls._sample_state_fidelities(
            n_samples=n_samples,
            random_key=random_key,
            model=model,
            kwargs=kwargs,
        )

        y: jnp.ndarray = jnp.linspace(0, 1, n_bins + 1)

        z, _ = jnp.histogram(fidelities, bins=y)

        z = z / n_samples

        return y, z

    @classmethod
    def _haar_probability(cls, fidelity: float, n_qubits: int) -> float:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876).

        Args:
            fidelity (float): fidelity of two parameter assignments in [0, 1]
            n_qubits (int): number of qubits in the quantum system

        Returns:
            float: probability for a given fidelity
        """
        N = 2**n_qubits

        prob = (N - 1) * (1 - fidelity) ** (N - 2)
        return prob

    @classmethod
    def _sample_haar_integral(cls, n_qubits: int, n_bins: int) -> jnp.ndarray:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
        into a 2D-histogram.

        Args:
            n_qubits (int): number of qubits in the quantum system
            n_bins (int): number of histogram bins

        Returns:
            jnp.ndarray: probability distribution for all fidelities
        """
        dist = np.zeros(n_bins)
        for idx in range(n_bins):
            v = idx / n_bins
            u = (idx + 1) / n_bins
            dist[idx], _ = integrate.quad(cls._haar_probability, v, u, args=(n_qubits,))

        return dist

    @classmethod
    def haar_integral(
        cls,
        n_qubits: int,
        n_bins: int,
        cache: bool = True,
        scale: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
        into a 3D-histogram.

        Args:
            n_qubits (int): number of qubits in the quantum system
            n_bins (int): number of histogram bins
            cache (bool): whether to cache the haar integral
            scale (bool): whether to scale the number of bins

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]:
                - x component (bins): the input domain
                - y component (probabilities): the haar probability density
                  funtion for random Haar states
        """
        if scale:
            n_bins = n_qubits * n_bins

        x = jnp.linspace(0, 1, n_bins)

        if cache:
            name = f"haar_{n_qubits}q_{n_bins}s_{'scaled' if scale else ''}.npy"

            cache_folder = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                y = jnp.load(file_path)
                return x, y

        y = cls._sample_haar_integral(n_qubits, n_bins)

        if cache:
            jnp.save(file_path, y)

        return x, y

    @classmethod
    def kullback_leibler_divergence(
        cls,
        vqc_prob_dist: jnp.ndarray,
        haar_dist: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Calculates the KL divergence between two probability distributions (Haar
        probability distribution and the fidelity distribution sampled from a VQC).

        Args:
            vqc_prob_dist (jnp.ndarray): VQC fidelity probability distribution.
                Should have shape (n_inputs_samples, n_bins)
            haar_dist (jnp.ndarray): Haar probability distribution with shape.
                Should have shape (n_bins, )

        Returns:
            jnp.ndarray: Array of KL-Divergence values for all values in axis 1
        """
        if len(vqc_prob_dist.shape) > 1:
            assert all([haar_dist.shape == p.shape for p in vqc_prob_dist]), (
                "All probabilities for inputs should have the same shape as Haar. "
                f"Got {haar_dist.shape} for Haar and {vqc_prob_dist.shape} for VQC"
            )
        else:
            vqc_prob_dist = vqc_prob_dist.reshape((1, -1))

        kl_divergence = np.zeros(vqc_prob_dist.shape[0])
        for idx, p in enumerate(vqc_prob_dist):
            kl_divergence[idx] = jnp.sum(rel_entr(p, haar_dist))

        return kl_divergence

    def kl_divergence_to_haar(
        model: Model,
        n_samples: int,
        n_bins: int,
        random_key: Optional[jax.random.PRNGKey] = None,
        scale: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        Shortcut method to compute the KL-Divergence bewteen a model and the
        Haar distribution. The basic steps are:
            - Sample the state fidelities for randomly initialised parameters.
            - Calculates the KL divergence between the sampled probability and
              the Haar probability distribution.

        Args:
            model (Model): Function that models the quantum circuit.
            n_samples (int): Number of parameter sets to generate.
            n_bins (int): Number of histogram bins.
            random_key (Optional[jax.random.PRNGKey]): JAX random key for
                parameter initialization. If None, uses the model's internal
                random key.
            scale (bool): Whether to scale the number of samples and bins.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Tuple containing the
                input samples, bin edges, and histogram values.
        """
        _, fidelities = Expressibility.state_fidelities(
            model=model,
            random_key=random_key,
            n_samples=n_samples,
            n_bins=n_bins,
            scale=scale,
            **kwargs,
        )
        _, haar_probs = Expressibility.haar_integral(
            model.n_qubits, n_bins=n_bins, scale=scale
        )
        return Expressibility.kullback_leibler_divergence(fidelities, haar_probs)
