# Ansaetze

.. or Ansatzes as preferred by the english community.
Anyway, we got various of the most-used Ansaetze implemented in this package.

You can load them manually by
```python
from qml_essentials.ansaetze import Ansaetze
print(Ansaetze.get_available())
```

However, usually you just want reference to them (by name) when instantiating a model.
To get an overview of all the available Ansaetze, checkout the [references](https://cirkiters.github.io/qml-essentials/references/).

If you want to implement your own ansatz, you can do so by inheriting from the `Circuit` class:
```python
from qml_essentials.ansaetze import Circuit

class MyHardwareEfficient(Circuit):
    @staticmethod
    def n_params_per_layer(n_qubits: int) -> int:
        return n_qubits * 3

    @staticmethod
    def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def build(w: np.ndarray, n_qubits: int):
        w_idx = 0
        for q in range(n_qubits):
            qml.RY(w[w_idx], wires=q)
            w_idx += 1
            qml.RZ(w[w_idx], wires=q)
            w_idx += 1

        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
```

and then pass it to the model:
```python
from qml_essentials.model import Model

model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type=MyHardwareEfficient,
)
```

Checkout page ["Usage"](usage.md) on how to proceed from here.