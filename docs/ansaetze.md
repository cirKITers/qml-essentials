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

For building the different Ansatzes, we use topology patterns defined in `qml_essentials.topologies`.
You can find a list of all the available topologies in the [references](https://cirkiters.github.io/qml-essentials/references/) as well.

To start implementing your own ansatz, you can inheriting from the `Circuit` class:
```python
from qml_essentials.ansaetze import Circuit
from qml_essentials.gates import Gates
from qml_essentials.topologies import Topology, Block

class MyHardwareEfficient(Circuit):
    @staticmethod
    def structure():
        return (
            Block(gate=Gates.RY),
            Block(gate=Gates.RZ),
            Block(gate=Gates.RY),
            Block(
                gate=Gates.CX,
                topology=Topology.stairs,
                stride=2,
                mirror=False,
            ),
            Block(
                gate=Gates.CX,
                topology=Topology.stairs,
                stride=2,
                offset=1,
                mirror=False,
            ),
        )

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

If you don't want to use the provided blocks and topologies, you can build your own Ansatz from scratch but have to implement all the required methods shown in the example below:
```python
import pennylane.numpy as np
from qml_essentials.gates import PulseInformation as pinfo
from typing import Optional

class MyHardwareEfficient(Circuit):
    @staticmethod
    def n_params_per_layer(n_qubits: int) -> int:
        return n_qubits * 3

    @staticmethod
    def n_pulse_params_per_layer(n_qubits: int) -> int:
        n_params_RY = pinfo.num_params("RY")
        n_params_RZ = pinfo.num_params("RZ")
        n_params_CZ = pinfo.num_params("CZ")

        n_pulse_params = (num_params_RY + num_params_RZ) * n_qubits
        n_pulse_params += num_params_CZ * (n_qubits - 1)

        return pulse_params

    @staticmethod
    def get_control_indices(n_qubits: int) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def build(w: np.ndarray, n_qubits: int, **kwargs):
        w_idx = 0
        for q in range(n_qubits):
            Gates.RY(w[w_idx], wires=q, **kwargs)
            w_idx += 1
            Gates.RZ(w[w_idx], wires=q, **kwargs)
            w_idx += 1

        if n_qubits > 1:
            for q in range(n_qubits - 1):
                Gates.CZ(wires=[q, q + 1], **kwargs)
```

The `**kwargs` allow both [noise simulation](noise.md) and [pulse simulation](pulses.md).
A custom `Circuit` should define `n_pulse_params_per_layer` if it will use pulse simulation at some point, but may be omitted otherwise.

Check out page [*Usage*](usage.md) on how to proceed from here.

## Custom Encoding

On model instantiation, you can choose how your inputs are encoded.
The default encoding is "RX" which will result in a single RX rotation per qubit.
You can change this behavior, by setting the optional `encoding` argument to
- a string or a list of strings where each is checked agains the [`Gates` class](https://cirkiters.github.io/qml-essentials/references/#gates)
- a callable or a list of callables

A callable must take an input, the wire where it's acting on and an optional noise_params dictionary.
Let's look at an example, where we want to encode a two-dimensional input:
```python
from qml_essentials.model import Model
from qml_essentials.gates import Gates

def MyCustomEncoding(w, wires, **kwars):
    Gates.RX(w[0], wires, **kwargs)
    Gates.RY(w[1], wires, **kwargs)

model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type=MyHardwareEfficient,
    encoding=MyCustomEncoding,
)

model(inputs=[1, 2])
```


## Overview

This section shows an overview of all the available Ansaetze in our package.
Most of the circuits are implemented according to to the original paper by [Sim et al.](https://doi.org/10.48550/arXiv.1905.10876).
*Note that Circuit 10 deviates from the original implementation!*

Oh and in case you need a refresh on the rotational axes and their corresponding states, here is a Bloch sphere :innocent: :

![Bloch Sphere](figures/bloch-sphere.svg#center)


### 4 Qubit Circuits
#### No Ansatz
![No Ansatz](figures/circuits_4q/No_Ansatz_light.png#circuit#only-light)
![No Ansatz](figures/circuits_4q/No_Ansatz_dark.png#circuit#only-dark)

#### Circuit 1
![Circuit 1](figures/circuits_4q/Circuit_1_light.png#circuit#only-light)
![Circuit 1](figures/circuits_4q/Circuit_1_dark.png#circuit#only-dark)

#### Circuit 2
![Circuit 2](figures/circuits_4q/Circuit_2_light.png#circuit#only-light)
![Circuit 2](figures/circuits_4q/Circuit_2_dark.png#circuit#only-dark)

#### Circuit 3
![Circuit 3](figures/circuits_4q/Circuit_3_light.png#circuit#only-light)
![Circuit 3](figures/circuits_4q/Circuit_3_dark.png#circuit#only-dark)

#### Circuit 4
![Circuit 4](figures/circuits_4q/Circuit_4_light.png#circuit#only-light)
![Circuit 4](figures/circuits_4q/Circuit_4_dark.png#circuit#only-dark)

#### Circuit 5
![Circuit 5](figures/circuits_4q/Circuit_5_light.png#circuit#only-light)
![Circuit 5](figures/circuits_4q/Circuit_5_dark.png#circuit#only-dark)

#### Circuit 6
![Circuit 6](figures/circuits_4q/Circuit_6_light.png#circuit#only-light)
![Circuit 6](figures/circuits_4q/Circuit_6_dark.png#circuit#only-dark)

#### Circuit 7
![Circuit 7](figures/circuits_4q/Circuit_7_light.png#circuit#only-light)
![Circuit 7](figures/circuits_4q/Circuit_7_dark.png#circuit#only-dark)

#### Circuit 8
![Circuit 8](figures/circuits_4q/Circuit_8_light.png#circuit#only-light)
![Circuit 8](figures/circuits_4q/Circuit_8_dark.png#circuit#only-dark)

#### Circuit 9
![Circuit 9](figures/circuits_4q/Circuit_9_light.png#circuit#only-light)
![Circuit 9](figures/circuits_4q/Circuit_9_dark.png#circuit#only-dark)

#### Circuit 10
![Circuit 10](figures/circuits_4q/Circuit_10_light.png#circuit#only-light)
![Circuit 10](figures/circuits_4q/Circuit_10_dark.png#circuit#only-dark)

#### Circuit 13
![Circuit 13](figures/circuits_4q/Circuit_13_light.png#circuit#only-light)
![Circuit 13](figures/circuits_4q/Circuit_13_dark.png#circuit#only-dark)

#### Circuit 14
![Circuit 14](figures/circuits_4q/Circuit_14_light.png#circuit#only-light)
![Circuit 14](figures/circuits_4q/Circuit_14_dark.png#circuit#only-dark)

#### Circuit 15
![Circuit 15](figures/circuits_4q/Circuit_15_light.png#circuit#only-light)
![Circuit 15](figures/circuits_4q/Circuit_15_dark.png#circuit#only-dark)

#### Circuit 16
![Circuit 16](figures/circuits_4q/Circuit_16_light.png#circuit#only-light)
![Circuit 16](figures/circuits_4q/Circuit_16_dark.png#circuit#only-dark)

#### Circuit 17
![Circuit 17](figures/circuits_4q/Circuit_17_light.png#circuit#only-light)
![Circuit 17](figures/circuits_4q/Circuit_17_dark.png#circuit#only-dark)

#### Circuit 18
![Circuit 18](figures/circuits_4q/Circuit_18_light.png#circuit#only-light)
![Circuit 18](figures/circuits_4q/Circuit_18_dark.png#circuit#only-dark)

#### Circuit 19
![Circuit 19](figures/circuits_4q/Circuit_19_light.png#circuit#only-light)
![Circuit 19](figures/circuits_4q/Circuit_19_dark.png#circuit#only-dark)

#### No Entangling
![No Entangling](figures/circuits_4q/No_Entangling_light.png#circuit#only-light)
![No Entangling](figures/circuits_4q/No_Entangling_dark.png#circuit#only-dark)

#### Strongly Entangling
![Strongly Entangling](figures/circuits_4q/Strongly_Entangling_light.png#circuit#only-light)
![Strongly Entangling](figures/circuits_4q/Strongly_Entangling_dark.png#circuit#only-dark)

#### Hardware Efficient
![Hardware Efficient](figures/circuits_4q/Hardware_Efficient_light.png#circuit#only-light)
![Hardware Efficient](figures/circuits_4q/Hardware_Efficient_dark.png#circuit#only-dark)

#### GHZ
![GHZ](figures/circuits_4q/GHZ_light.png#circuit#only-light)
![GHZ](figures/circuits_4q/GHZ_dark.png#circuit#only-dark)


### 5 Qubit Circuits
#### No Ansatz
![No Ansatz](figures/circuits_5q/No_Ansatz_light.png#circuit#only-light)
![No Ansatz](figures/circuits_5q/No_Ansatz_dark.png#circuit#only-dark)

#### Circuit 1
![Circuit 1](figures/circuits_5q/Circuit_1_light.png#circuit#only-light)
![Circuit 1](figures/circuits_5q/Circuit_1_dark.png#circuit#only-dark)

#### Circuit 2
![Circuit 2](figures/circuits_5q/Circuit_2_light.png#circuit#only-light)
![Circuit 2](figures/circuits_5q/Circuit_2_dark.png#circuit#only-dark)

#### Circuit 3
![Circuit 3](figures/circuits_5q/Circuit_3_light.png#circuit#only-light)
![Circuit 3](figures/circuits_5q/Circuit_3_dark.png#circuit#only-dark)

#### Circuit 4
![Circuit 4](figures/circuits_5q/Circuit_4_light.png#circuit#only-light)
![Circuit 4](figures/circuits_5q/Circuit_4_dark.png#circuit#only-dark)

#### Circuit 5
![Circuit 5](figures/circuits_5q/Circuit_5_light.png#circuit#only-light)
![Circuit 5](figures/circuits_5q/Circuit_5_dark.png#circuit#only-dark)

#### Circuit 6
![Circuit 6](figures/circuits_5q/Circuit_6_light.png#circuit#only-light)
![Circuit 6](figures/circuits_5q/Circuit_6_dark.png#circuit#only-dark)

#### Circuit 7
![Circuit 7](figures/circuits_5q/Circuit_7_light.png#circuit#only-light)
![Circuit 7](figures/circuits_5q/Circuit_7_dark.png#circuit#only-dark)

#### Circuit 8
![Circuit 8](figures/circuits_5q/Circuit_8_light.png#circuit#only-light)
![Circuit 8](figures/circuits_5q/Circuit_8_dark.png#circuit#only-dark)

#### Circuit 9
![Circuit 9](figures/circuits_5q/Circuit_9_light.png#circuit#only-light)
![Circuit 9](figures/circuits_5q/Circuit_9_dark.png#circuit#only-dark)

#### Circuit 10
![Circuit 10](figures/circuits_5q/Circuit_10_light.png#circuit#only-light)
![Circuit 10](figures/circuits_5q/Circuit_10_dark.png#circuit#only-dark)

#### Circuit 13
![Circuit 13](figures/circuits_5q/Circuit_13_light.png#circuit#only-light)
![Circuit 13](figures/circuits_5q/Circuit_13_dark.png#circuit#only-dark)

#### Circuit 14
![Circuit 14](figures/circuits_5q/Circuit_14_light.png#circuit#only-light)
![Circuit 14](figures/circuits_5q/Circuit_14_dark.png#circuit#only-dark)

#### Circuit 15
![Circuit 15](figures/circuits_5q/Circuit_15_light.png#circuit#only-light)
![Circuit 15](figures/circuits_5q/Circuit_15_dark.png#circuit#only-dark)

#### Circuit 16
![Circuit 16](figures/circuits_5q/Circuit_16_light.png#circuit#only-light)
![Circuit 16](figures/circuits_5q/Circuit_16_dark.png#circuit#only-dark)

#### Circuit 17
![Circuit 17](figures/circuits_5q/Circuit_17_light.png#circuit#only-light)
![Circuit 17](figures/circuits_5q/Circuit_17_dark.png#circuit#only-dark)

#### Circuit 18
![Circuit 18](figures/circuits_5q/Circuit_18_light.png#circuit#only-light)
![Circuit 18](figures/circuits_5q/Circuit_18_dark.png#circuit#only-dark)

#### Circuit 19
![Circuit 19](figures/circuits_5q/Circuit_19_light.png#circuit#only-light)
![Circuit 19](figures/circuits_5q/Circuit_19_dark.png#circuit#only-dark)

#### No Entangling
![No Entangling](figures/circuits_5q/No_Entangling_light.png#circuit#only-light)
![No Entangling](figures/circuits_5q/No_Entangling_dark.png#circuit#only-dark)

#### Strongly Entangling
![Strongly Entangling](figures/circuits_5q/Strongly_Entangling_light.png#circuit#only-light)
![Strongly Entangling](figures/circuits_5q/Strongly_Entangling_dark.png#circuit#only-dark)

#### Hardware Efficient
![Hardware Efficient](figures/circuits_5q/Hardware_Efficient_light.png#circuit#only-light)
![Hardware Efficient](figures/circuits_5q/Hardware_Efficient_dark.png#circuit#only-dark)

#### GHZ
![GHZ](figures/circuits_5q/GHZ_light.png#circuit#only-light)
![GHZ](figures/circuits_5q/GHZ_dark.png#circuit#only-dark)


### 6 Qubit Circuits
#### No Ansatz
![No Ansatz](figures/circuits_6q/No_Ansatz_light.png#circuit#only-light)
![No Ansatz](figures/circuits_6q/No_Ansatz_dark.png#circuit#only-dark)

#### Circuit 1
![Circuit 1](figures/circuits_6q/Circuit_1_light.png#circuit#only-light)
![Circuit 1](figures/circuits_6q/Circuit_1_dark.png#circuit#only-dark)

#### Circuit 2
![Circuit 2](figures/circuits_6q/Circuit_2_light.png#circuit#only-light)
![Circuit 2](figures/circuits_6q/Circuit_2_dark.png#circuit#only-dark)

#### Circuit 3
![Circuit 3](figures/circuits_6q/Circuit_3_light.png#circuit#only-light)
![Circuit 3](figures/circuits_6q/Circuit_3_dark.png#circuit#only-dark)

#### Circuit 4
![Circuit 4](figures/circuits_6q/Circuit_4_light.png#circuit#only-light)
![Circuit 4](figures/circuits_6q/Circuit_4_dark.png#circuit#only-dark)

#### Circuit 5
![Circuit 5](figures/circuits_6q/Circuit_5_light.png#circuit#only-light)
![Circuit 5](figures/circuits_6q/Circuit_5_dark.png#circuit#only-dark)

#### Circuit 6
![Circuit 6](figures/circuits_6q/Circuit_6_light.png#circuit#only-light)
![Circuit 6](figures/circuits_6q/Circuit_6_dark.png#circuit#only-dark)

#### Circuit 7
![Circuit 7](figures/circuits_6q/Circuit_7_light.png#circuit#only-light)
![Circuit 7](figures/circuits_6q/Circuit_7_dark.png#circuit#only-dark)

#### Circuit 8
![Circuit 8](figures/circuits_6q/Circuit_8_light.png#circuit#only-light)
![Circuit 8](figures/circuits_6q/Circuit_8_dark.png#circuit#only-dark)

#### Circuit 9
![Circuit 9](figures/circuits_6q/Circuit_9_light.png#circuit#only-light)
![Circuit 9](figures/circuits_6q/Circuit_9_dark.png#circuit#only-dark)

#### Circuit 10
![Circuit 10](figures/circuits_6q/Circuit_10_light.png#circuit#only-light)
![Circuit 10](figures/circuits_6q/Circuit_10_dark.png#circuit#only-dark)

#### Circuit 13
![Circuit 13](figures/circuits_6q/Circuit_13_light.png#circuit#only-light)
![Circuit 13](figures/circuits_6q/Circuit_13_dark.png#circuit#only-dark)

#### Circuit 14
![Circuit 14](figures/circuits_6q/Circuit_14_light.png#circuit#only-light)
![Circuit 14](figures/circuits_6q/Circuit_14_dark.png#circuit#only-dark)

#### Circuit 15
![Circuit 15](figures/circuits_6q/Circuit_15_light.png#circuit#only-light)
![Circuit 15](figures/circuits_6q/Circuit_15_dark.png#circuit#only-dark)

#### Circuit 16
![Circuit 16](figures/circuits_6q/Circuit_16_light.png#circuit#only-light)
![Circuit 16](figures/circuits_6q/Circuit_16_dark.png#circuit#only-dark)

#### Circuit 17
![Circuit 17](figures/circuits_6q/Circuit_17_light.png#circuit#only-light)
![Circuit 17](figures/circuits_6q/Circuit_17_dark.png#circuit#only-dark)

#### Circuit 18
![Circuit 18](figures/circuits_6q/Circuit_18_light.png#circuit#only-light)
![Circuit 18](figures/circuits_6q/Circuit_18_dark.png#circuit#only-dark)

#### Circuit 19
![Circuit 19](figures/circuits_6q/Circuit_19_light.png#circuit#only-light)
![Circuit 19](figures/circuits_6q/Circuit_19_dark.png#circuit#only-dark)

#### No Entangling
![No Entangling](figures/circuits_6q/No_Entangling_light.png#circuit#only-light)
![No Entangling](figures/circuits_6q/No_Entangling_dark.png#circuit#only-dark)

#### Strongly Entangling
![Strongly Entangling](figures/circuits_6q/Strongly_Entangling_light.png#circuit#only-light)
![Strongly Entangling](figures/circuits_6q/Strongly_Entangling_dark.png#circuit#only-dark)

#### Hardware Efficient
![Hardware Efficient](figures/circuits_6q/Hardware_Efficient_light.png#circuit#only-light)
![Hardware Efficient](figures/circuits_6q/Hardware_Efficient_dark.png#circuit#only-dark)

#### GHZ
![GHZ](figures/circuits_6q/GHZ_light.png#circuit#only-light)
![GHZ](figures/circuits_6q/GHZ_dark.png#circuit#only-dark)

