# Ansaetze

.. or Ansatzes as preferred by the english community.
Anyway, we got various of the most-used Ansaetze implemented in this package. :rocket:

You can load them manually by
```python
from qml_essentials.ansaetze import Ansaetze
all_ansaetze = Ansaetze.get_available()

for ansatz in all_ansaetze:
    print(ansatz.__name__)
```

See the [*Overview*](#overview) at the end of this document for more details.
However, usually you just want reference to them (by name) when instantiating a model.
To get an overview of all the available Ansaetze, checkout the [references](https://cirkiters.github.io/qml-essentials/references/).

## Custom Ansatz

If you want to implement your own ansatz, you can do so by inheriting from the `Circuit` class:
```python
import pennylane as qml
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
    "Depolarizing": 0.0,
    "MultiQubitDepolarizing": 0.0,
}
```

Providing this optional input will apply the corresponding noise to the model where the Bit Flip, Phase Flip, Depolarizing and Two-Qubit Depolarizing Channels are applied after each gate and the Amplitude and Phase Damping are applied at the end of the circuit.
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
        "Depolarizing": 0.05,
        "MultiQubitDepolarizing": 0.06
})
```

## Overview

This section shows an overview of all the available Ansaetze in our package.
Most of the circuits are implemented according to to the original paper by [Sim et al.](https://doi.org/10.48550/arXiv.1905.10876).
*Note that Circuit 10 deviates from the original implementation!*

Oh and in case you need a refresh on the rotational axes and their corresponding states, here is a Bloch sphere :innocent: :

![Bloch Sphere](figures/bloch-sphere.svg#center)

### No Ansatz
![No Ansatz](figures/No_Ansatz_light.png#circuit#only-light)
![No Ansatz](figures/No_Ansatz_dark.png#circuit#only-dark)

### Circuit 1
![Circuit 1](figures/Circuit_1_light.png#circuit#only-light)
![Circuit 1](figures/Circuit_1_dark.png#circuit#only-dark)

### Circuit 2
![Circuit 2](figures/Circuit_2_light.png#circuit#only-light)
![Circuit 2](figures/Circuit_2_dark.png#circuit#only-dark)

### Circuit 3
![Circuit 3](figures/Circuit_3_light.png#circuit#only-light)
![Circuit 3](figures/Circuit_3_dark.png#circuit#only-dark)

### Circuit 4
![Circuit 4](figures/Circuit_4_light.png#circuit#only-light)
![Circuit 4](figures/Circuit_4_dark.png#circuit#only-dark)

### Circuit 6
![Circuit 6](figures/Circuit_6_light.png#circuit#only-light)
![Circuit 6](figures/Circuit_6_dark.png#circuit#only-dark)

### Circuit 9
![Circuit 9](figures/Circuit_9_light.png#circuit#only-light)
![Circuit 9](figures/Circuit_9_dark.png#circuit#only-dark)

### Circuit 10
![Circuit 10](figures/Circuit_10_light.png#circuit#only-light)
![Circuit 10](figures/Circuit_10_dark.png#circuit#only-dark)

### Circuit 15
![Circuit 15](figures/Circuit_15_light.png#circuit#only-light)
![Circuit 15](figures/Circuit_15_dark.png#circuit#only-dark)

### Circuit 16
![Circuit 16](figures/Circuit_16_light.png#circuit#only-light)
![Circuit 16](figures/Circuit_16_dark.png#circuit#only-dark)

### Circuit 17
![Circuit 17](figures/Circuit_17_light.png#circuit#only-light)
![Circuit 17](figures/Circuit_17_dark.png#circuit#only-dark)

### Circuit 18
![Circuit 18](figures/Circuit_18_light.png#circuit#only-light)
![Circuit 18](figures/Circuit_18_dark.png#circuit#only-dark)

### Circuit 19
![Circuit 19](figures/Circuit_19_light.png#circuit#only-light)
![Circuit 19](figures/Circuit_19_dark.png#circuit#only-dark)

### No Entangling
![No Entangling](figures/No_Entangling_light.png#circuit#only-light)
![No Entangling](figures/No_Entangling_dark.png#circuit#only-dark)

### Strongly Entangling
![Strongly Entangling](figures/Strongly_Entangling_light.png#circuit#only-light)
![Strongly Entangling](figures/Strongly_Entangling_dark.png#circuit#only-dark)

### Hardware Efficient
![Hardware Efficient](figures/Hardware_Efficient_light.png#circuit#only-light)
![Hardware Efficient](figures/Hardware_Efficient_dark.png#circuit#only-dark)

### GHZ
![GHZ](figures/GHZ_light.png#circuit#only-light)
![GHZ](figures/GHZ_dark.png#circuit#only-dark)

