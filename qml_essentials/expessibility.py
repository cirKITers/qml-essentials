import pennylane.numpy as np
from typing import Tuple, Callable, Any
from scipy import integrate
from scipy.special import rel_entr
import os


class Expressibility:
    def __init__(
        self,
        model: Callable,  # type: ignore
        n_samples: int = 1000,
        n_input_samples: int = 10,
        seed: int = 100,
        **kwargs: Any,
    ) -> None:

        self.model = model
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

        self.epsilon = 1e-5

        x_domain = [-1 * np.pi, 1 * np.pi]
        self.x_samples = np.linspace(
            x_domain[0], x_domain[1], n_input_samples, requires_grad=False
        )

        self.kwargs = kwargs

    def _sample_state_fidelities(self) -> np.ndarray:
        n_x_samples = len(self.x_samples)

        fidelities = np.zeros((n_x_samples, self.n_samples))

        w = (
            2
            * np.pi
            * (
                1
                - 2
                * self.rng.random(size=[*self.model.params.shape, self.n_samples * 2])
            )
        )

        # TODO: Maybe we could vectorize this for loop as follows ?
        # x_samples_batched = self.x_samples.repeat(self.n_samples * 2)
        # w_batched = w.repeat(n_x_samples, axis=2)
        # self.model(inputs=x_samples_batched, params=w_batched, **self.kwargs)

        x_samples_batched = self.x_samples.reshape(1, -1).repeat(
            self.n_samples * 2, axis=0
        )

        for idx in range(n_x_samples):

            sv = self.model(inputs=x_samples_batched[:, idx], params=w, **self.kwargs)
            sqrt_sv1 = np.sqrt(sv[: self.n_samples])

            fidelity = (
                np.trace(
                    np.sqrt(sqrt_sv1 * sv[self.n_samples :] * sqrt_sv1),
                    axis1=1,
                    axis2=2,
                )
                ** 2
            )
            fidelities[idx] = np.real(fidelity)

        return fidelities

    def sample_hist_state_fidelities(
        self, n_bins: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fidelities = self._sample_state_fidelities()
        z_component = np.zeros((len(self.x_samples), n_bins))

        # FIXME: somehow I get nan's in the histogram, when directly creating bins until n
        # workaround hack is to add a small epsilon
        # could it be related to sampling issues?
        b = np.linspace(0, 1 + self.epsilon, n_bins + 1)
        for i, f in enumerate(fidelities):
            z_component[i], _ = np.histogram(f, bins=b)
        z_component = z_component / self.n_samples
        return self.x_samples, b, z_component

    @staticmethod
    def theoretical_haar_probability(fidelity: float, n_qubits: int) -> float:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876).

        Parameters
        ----------
        fidelity : float
            fidelity of two parameter assignments in [0, 1]
        n_qubits : int
            number of qubits in the quantum system

        Returns
        -------
        float
            probability for a given fidelity
        """
        N = 2**n_qubits

        prob = (N - 1) * (1 - fidelity) ** (N - 2)
        return prob

    @staticmethod
    def sampled_haar_probability(n_qubits: int, n_bins: int) -> np.ndarray:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
        into a 2D-histogram.

        Parameters
        ----------
        n_qubits : int
            number of qubits in the quantum system
        n_bins : int
            number of histogram bins

        Returns
        -------
        np.ndarray
            probability distribution for all fidelities
        """
        dist = np.zeros(n_bins)
        for i in range(n_bins):
            l = (1 / n_bins) * i
            u = l + (1 / n_bins)
            dist[i], _ = integrate.quad(
                Expressibility.theoretical_haar_probability, l, u, args=(n_qubits,)
            )

        return dist

    @staticmethod
    def get_sampled_haar_probability_histogram(
        n_qubits: int,
        n_bins: int,
        cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates theoretical probability density function for random Haar states
        as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
        into a 3D-histogram.

        Parameters
        ----------
        n_qubits : int
            number of qubits in the quantum system
        n_bins : int
            number of histogram bins
        cache : bool
            [TODO:description]

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            [TODO:description]
            - x component (bins)
            - y component (probabilities)
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

        # Note that this is a jax rng, so it does not matter if we
        # call that multiple times
        y = Expressibility.sampled_haar_probability(n_qubits, n_bins)

        if cache:
            np.save(file_path, y)

        return x, y

    @staticmethod
    def get_kl_divergence_expr(
        vqc_prob_dist: np.ndarray,
        haar_dist: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the KL divergence between two probability distributions (Haar
        probability distribution and the fidelity distribution sampled from a VQC).

        Parameters
        ----------
        vqc_prob_dist : np.ndarray
            VQC fidelity probability distribution. Should have shape
            (n_inputs_samples, n_bins)
        haar_dist : np.ndarray
            Haar probability distribution with shape. Should have shape (n_bins, )

        Returns
        -------
        np.ndarray
            Array of KL-Divergence values for all values in axis 1
        """
        assert all([haar_dist.shape == p.shape for p in vqc_prob_dist]), (
            "All "
            "probabilities for inputs should have the same shape as Haar. "
            f"Got {haar_dist.shape} for Haar and {vqc_prob_dist.shape} for VQC"
        )

        kl_divergence = np.zeros(vqc_prob_dist.shape[0])
        for i, p in enumerate(vqc_prob_dist):
            kl_divergence[i] = np.sum(rel_entr(p, haar_dist))

        return kl_divergence
