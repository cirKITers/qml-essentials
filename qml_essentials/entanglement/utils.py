import logging
import numpy as np
import pennylane as qml

log = logging.getLogger(__name__)

class EntanglementMeasure:
    def __init__(self, name=None, allowNoise=False, requiresDevice=False, requiresModel=False):
        self.name = name
        self.allowNoise = allowNoise

        self.additionalArgs = {}
        if requiresDevice:
            self.additionalArgs["getDevice"] = lambda **args: qml.device(**args)

        self.requiresModel = requiresModel

    def __call__(self, measure):
        def wrapper(model, n_samples, seed, scale=False, strategy=None, **kwargs):
            if not self.allowNoise and "noise_params" in kwargs:
                log.warning(f"{self.name} measure not suitable for noisy circuits.")

            n = model.n_qubits
            N = 2 ** n

            if self.requiresModel:
                self.additionalArgs["model"] = model

            if strategy:
                self.additionalArgs["strategy"] = strategy

            # Implicitly set input to none in case it's not needed
            kwargs.setdefault("inputs", None)
            rhos = model(execution_type="density", **kwargs)
            rhos = rhos.reshape(-1, N, N)

            entanglement = np.zeros(len(rhos))

            for i, rho in enumerate(rhos):
                entanglement[i] = measure(rho, n, **self.additionalArgs)

            # Catch floating point errors
            entangling_capability = min(max(entanglement.mean(), 0.0), 1.0)

            log.debug(f"Variance of measure: {entanglement.var()}")
            return float(entangling_capability)

        return wrapper

class ConvexRoofExtension:
    def __call__(self, measure):
        def wrapper(rho, n, strategy=eigen_decomp_strategy, **kwargs):
            return strategy(measure, rho, n, **kwargs)

        return wrapper


def eigen_decomp_strategy(measure, rho, n, **kwargs):
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    entanglement = 0
    for prob, ev in zip(eigenvalues, eigenvectors):
        ev = ev.reshape(-1, 1)
        sigma = ev @ np.conjugate(ev).T
        entanglement += prob * measure(sigma, n, **kwargs)
    return entanglement


