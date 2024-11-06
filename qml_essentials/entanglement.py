from typing import Optional, Any
import pennylane as qml
import pennylane.numpy as np

from qml_essentials.model import Model
import logging

log = logging.getLogger(__name__)


class Entanglement:

    @staticmethod
    def meyer_wallach(
        model: Model, n_samples: Optional[int | None], seed: Optional[int], **kwargs: Any  # type: ignore
    ) -> float:
        """
        Calculates the entangling capacity of a given quantum circuit
        using Meyer-Wallach measure.

        Args:
            model (Callable): Function that models the quantum circuit.
            n_samples (int): Number of samples per qubit.
                If None or < 0, the current parameters of the model are used
            seed (Optional[int]): Seed for the random number generator.
            kwargs (Any): Additional keyword arguments for the model function.

        Returns:
            float: Entangling capacity of the given circuit. It is guaranteed
                to be between 0.0 and 1.0.
        """
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
        mw_measure = np.zeros(n_samples, dtype=complex)
        qb = list(range(model.n_qubits))

        # TODO: vectorize in future iterations
        for i in range(n_samples):
            # implicitly set input to none in case it's not needed
            kwargs.setdefault("inputs", None)
            # explicitly set execution type because everything else won't work
            U = model(params=params[:, :, i], execution_type="density", **kwargs)

            entropy = 0

            for j in range(model.n_qubits):
                density = qml.math.partial_trace(U, qb[:j] + qb[j + 1 :])
                entropy += np.trace((density @ density).real)

            mw_measure[i] = 1 - entropy / model.n_qubits

        mw = 2 * np.sum(mw_measure.real) / n_samples

        # catch floating point errors
        entangling_capability = min(max(mw, 0.0), 1.0)

        return float(entangling_capability)
