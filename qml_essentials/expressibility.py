import pennylane.numpy as np
from typing import Tuple, List, Any
from scipy import integrate
from scipy.special import rel_entr
import os

from qml_essentials.model import Model


class Expressibility:
    @staticmethod
    def _sample_state_fidelities(
        model: Model,
        x_samples: np.ndarray,
        n_samples: int,
        seed: int,
        kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the fidelities for each pair of input samples and parameter sets.

        Args:
            model (Callable): Function that models the quantum circuit.
            x_samples (np.ndarray): Array of shape (n_input_samples, n_features)
                containing the input samples.
            n_samples (int): Number of parameter sets to generate.
            seed (int): Random number generator seed.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            np.ndarray: Array of shape (n_input_samples, n_samples)
            containing the fidelities.
        """
        rng = np.random.default_rng(seed)

        # Number of input samples
        n_x_samples = len(x_samples)

        # Initialize array to store fidelities
        fidelities: np.ndarray = np.zeros((n_x_samples, n_samples))

        # Generate random parameter sets
        # We need two sets of parameters, as we are computing fidelities for a
        # pair of random state vectors
        model.initialize_params(rng=rng, repeat=n_samples * 2)
        # Batch input samples and parameter sets for efficient computation
        x_samples_batched: np.ndarray = x_samples.reshape(1, -1).repeat(
            n_samples * 2, axis=0
        )

        # Compute the fidelity for each pair of input samples and parameters
        for idx in range(n_x_samples):

            # Evaluate the model for the current pair of input samples and parameters
            # Execution type is explicitly set to density
            sv: np.ndarray = model(
                inputs=x_samples_batched[:, idx],
                params=model.params,
                execution_type="density",
                **kwargs,
            )
            sqrt_sv1: np.ndarray = np.sqrt(sv[:n_samples])

            # Compute the fidelity using the partial trace of the statevector
            fidelity: np.ndarray = (
                np.trace(
                    np.sqrt(sqrt_sv1 * sv[n_samples:] * sqrt_sv1),
                    axis1=1,
                    axis2=2,
                )
                ** 2
            )
            fidelities[idx] = np.real(fidelity)

        return fidelities

    @staticmethod
    def state_fidelities(
        seed: int,
        n_samples: int,
        n_bins: int,
        n_input_samples: int,
        input_domain: List[float],
        model: Model,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the state fidelities and histogram them into a 2D array.

        Args:
            seed (int): Random number generator seed.
            n_samples (int): Number of parameter sets to generate.
            n_bins (int): Number of histogram bins.
            n_input_samples (int): Number of input samples.
            input_domain (List[float]): Input domain.
            model (Callable): Function that models the quantum circuit.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the
                input samples, bin edges, and histogram values.
        """

        x = np.linspace(*input_domain, n_input_samples, requires_grad=False)

        fidelities = Expressibility._sample_state_fidelities(
            x_samples=x,
            n_samples=n_samples,
            seed=seed,
            model=model,
            kwargs=kwargs,
        )
        z: np.ndarray = np.zeros((n_input_samples, n_bins))

        y: np.ndarray = np.linspace(0, 1, n_bins + 1)

        for i, f in enumerate(fidelities):
            z[i], _ = np.histogram(f, bins=y)

        z = z / n_samples

        return x, y, z

    @staticmethod
    def _haar_probability(fidelity: float, n_qubits: int) -> float:
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

    @staticmethod
    def _sample_haar_integral(n_qubits: int, n_bins: int) -> np.ndarray:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
        into a 2D-histogram.

        Args:
            n_qubits (int): number of qubits in the quantum system
            n_bins (int): number of histogram bins

        Returns:
            np.ndarray: probability distribution for all fidelities
        """
        dist = np.zeros(n_bins)
        for idx in range(n_bins):
            v = (1 / n_bins) * idx
            u = v + (1 / n_bins)
            dist[idx], _ = integrate.quad(
                Expressibility._haar_probability, v, u, args=(n_qubits,)
            )

        return dist

    @staticmethod
    def haar_integral(
        n_qubits: int,
        n_bins: int,
        cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
        into a 3D-histogram.

        Args:
            n_qubits (int): number of qubits in the quantum system
            n_bins (int): number of histogram bins
            cache (bool): whether to cache the haar integral

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - x component (bins): the input domain
                - y component (probabilities): the haar probability density
                  funtion for random Haar states
        """
        x = np.linspace(0, 1, n_bins)

        if cache:
            name = f"haar_{n_qubits}q_{n_bins}s.npy"

            cache_folder = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                y = np.load(file_path)
                return x, y

        y = Expressibility._sample_haar_integral(n_qubits, n_bins)

        if cache:
            np.save(file_path, y)

        return x, y

    @staticmethod
    def kullback_leibler_divergence(
        vqc_prob_dist: np.ndarray,
        haar_dist: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the KL divergence between two probability distributions (Haar
        probability distribution and the fidelity distribution sampled from a VQC).

        Args:
            vqc_prob_dist (np.ndarray): VQC fidelity probability distribution.
                Should have shape (n_inputs_samples, n_bins)
            haar_dist (np.ndarray): Haar probability distribution with shape.
                Should have shape (n_bins, )

        Returns:
            np.ndarray: Array of KL-Divergence values for all values in axis 1
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
            kl_divergence[idx] = np.sum(rel_entr(p, haar_dist))

        return kl_divergence
