from qml_essentials.model import Model
from qml_essentials.entanglement import Entanglement

noise = {"Depolarizing": 0.02}
# noise={}

model = Model(
    n_qubits=4,
    n_layers=1,
    circuit_type="Hardware_Efficient",
    output_qubit=-1,
)

ent = Entanglement.concentratable_entanglement(
    model, n_samples=250, seed=1000, noise_params=noise
)
print(ent)
