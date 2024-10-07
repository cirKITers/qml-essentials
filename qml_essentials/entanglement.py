from typing import Optional, Any
import pennylane as qml
import pennylane.numpy as np

from qml_essentials.model import Model
import logging

log = logging.getLogger(__name__)


class Entanglement:

    @staticmethod
    def meyer_wallach(
        model: Model, n_samples: int, seed: Optional[int], **kwargs: Any  # type: ignore
    ) -> float:
        """
        Calculates the entangling capacity of a given quantum circuit
        using Meyer-Wallach measure.

        Args:
            model (Callable): Function that models the quantum circuit.
            n_samples (int): Number of samples per qubit.
            seed (Optional[int]): Seed for the random number generator.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Entangling capacity of the given circuit. It is guaranteed
                to be between 0.0 and 1.0.
        """
        rng = np.random.default_rng(seed)
        if n_samples > 0:
            assert seed is not None, "Seed must be provided when samples > 0"
            # TODO: maybe switch to JAX rng
            # params = rng.uniform(0, 2 * np.pi, size=(n_samples, *model.params.shape))
            model.initialize_params(rng=rng, repeat=n_samples)
        else:
            if seed is not None:
                log.warning("Seed is ignored when samples is 0")
            n_samples = 1
            model.initialize_params(rng=rng, repeat=1)

        samples = model.params.shape[-1]
        mw_measure = np.zeros(samples, dtype=complex)
        qb = list(range(model.n_qubits))

        # TODO: vectorize in future iterations
        for i in range(samples):
            # implicitly set input to none in case it's not needed
            kwargs.setdefault("inputs", None)
            # explicitly set execution type because everything else won't work
            U = model(params=model.params[:, :, i], execution_type="density", **kwargs)

            entropy = 0

            for j in range(model.n_qubits):
                density = qml.math.partial_trace(U, qb[:j] + qb[j + 1 :])
                entropy += np.trace((density @ density).real)

            mw_measure[i] = 1 - entropy / model.n_qubits

        mw = 2 * np.sum(mw_measure.real) / samples

        # catch floating point errors
        entangling_capability = min(max(mw, 0.0), 1.0)

        return float(entangling_capability)
