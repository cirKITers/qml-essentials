# Ansaetze

.. or Ansatzes as preferred by the english community.
Anyway, we got various of the most-used Ansaetze implemented in this package.

You can load them manually by
```python
from qml_essentials.ansaetze import Ansaetze
all_ansaetze = Ansaetze.get_available()

for ansatz in all_ansaetze:
    print(ansatz.__name__)
```

```
No_Ansatz
Circuit_1
Circuit_2
Circuit_3
Circuit_4
Circuit_6
Circuit_9
Circuit_10
Circuit_15
Circuit_16
Circuit_17
Circuit_18
Circuit_19
No_Entangling
Strongly_Entangling
Hardware_Efficient
```

*Note that Circuit 10 deviates from the original implementation!*

However, usually you just want reference to them (by name) when instantiating a model.
To get an overview of all the available Ansaetze, checkout the [references](https://cirkiters.github.io/qml-essentials/references/).

## Custom Ansatz

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
    def build(w: np.ndarray, n_qubits: int, noise_params=None):
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

Checkout page [*Usage*](usage.md) on how to proceed from here.

## Custom Encoding

On model instantiation, you can choose how your inputs are encoded.
The default encoding is "RX" which will result in a single RX rotation per qubit.
You can change this behavior, by setting the optional `encoding` argument to
- a string or a list of strings where each is checked agains the [`Gates` class](https://cirkiters.github.io/qml-essentials/references/#gates)
- a callable or a list of callables

A callable must take an input, the wire where it's acting on and an optional noise_params dictionary.
Let's look at an example, where we wan't to encode a two-dimensional input:
```python
from qml_essentials.model import Model
from qml_essentials.ansaetze import Gates

def MyCustomEncoding(w, wires, noise_params=None):
    Gates.RX(w[0], wires, noise_params=noise_params)
    Gates.RY(w[1], wires, noise_params=noise_params)

model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type=MyHardwareEfficient,
    encoding=MyCustomEncoding,
)

model(inputs=[1, 2])
```


## Noise

You might have noticed, that the `build` method takes an additional input `noise_params`, which we did not used so far.
In general, all of the Ansatzes, that are implemented in this package allow this additional input which is a dictionary containing all the noise parameters of the circuit (here all with probability $0.0$):
```python
noise_params = {
    "BitFlip": 0.0,
    "PhaseFlip": 0.0,
    "AmplitudeDamping": 0.0,
    "PhaseDamping": 0.0,
    "DepolarizingChannel": 0.0,
}
```

Providing this optional input will apply the corresponding noise to the model where the Bit Flip, Phase Flip and Depolarizing Channel are applied after each gate and the Amplitude and Phase Damping are applied at the end of the circuit.
To achieve this, we implement our own set of noisy gates, that build upon the Pennylane gates. To demonstrate this, let's extend our example above:
```python
from qml_essentials.ansaetze import Gates, Circuit

class MyNoisyHardwareEfficient(Circuit):
    @staticmethod
    def n_params_per_layer(n_qubits: int) -> int:
        return n_qubits * 3

    @staticmethod
    def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def build(w: np.ndarray, n_qubits: int, noise_params=None):
        w_idx = 0
        for q in range(n_qubits):
            Gates.RY(w[w_idx], wires=q, noise_params=noise_params)
            w_idx += 1
            Gates.RZ(w[w_idx], wires=q, noise_params=noise_params)
            w_idx += 1

        if n_qubits > 1:
            for q in range(n_qubits - 1):
                Gates.CZ(wires=[q, q + 1], noise_params=noise_params)
```

As you can see, we slightly modified the example, by importing the `Gates` class from `ansaetze` and by adding the `noise_params` input to each of the gates.
When using a noisy circuit, make sure to run the model with the `density` execution type:
```python
model(
    model.params,
    inputs=None,
    execution_type="density",
    noise_params={
        "BitFlip": 0.01,
        "PhaseFlip": 0.02,
        "AmplitudeDamping": 0.03,
        "PhaseDamping": 0.04,
        "DepolarizingChannel": 0.05,
})
```