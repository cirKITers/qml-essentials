from qml_essentials.entanglement import concentratable_entanglement
from qml_essentials.model import Model

if __name__ == "__main__":
    n_qubits = 2
    seed = 1000
    n_layers = 1

    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type="Circuit_1"
    )

    print(concentratable_entanglement(model=model, n_samples=3, seed=seed, noise_params={"Depolarizing": 0.04}))