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
import pennylane.numpy as np
from qml_essentials.ansaetze import Circuit
from qml_essentials.ansaetze import PulseInformation as pinfo
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
            qml.RY(w[w_idx], wires=q, **kwargs)
            w_idx += 1
            qml.RZ(w[w_idx], wires=q, **kwargs)
            w_idx += 1

        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1], **kwargs)
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
The `**kwargs` allow both [noise simulation](#noise) and [pulse simulation](#pulse-simulation).
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
from qml_essentials.ansaetze import Gates

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

## Noise
You might have noticed, that the `build` method takes the additional input **kwargs, which we did not used so far.
In general, all of the Ansatzes that are implemented in this package allow the additional input below which is a dictionary containing all the noise parameters of the circuit (here all with probability $0.0$):
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

To demonstrate this, let's recall the custom ansatz `MyHardwareEfficient` defined in [Custom Ansatz](#custom-ansatz) and extend the model's usage:

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

In addition to these decoherent errors, we can also apply a `GateError` which affects each parameterized gate as $w = w + \mathcal{N}(0, \epsilon)$, where $\sqrt{\epsilon}$ is the standard deviation of the noise, specified by the `GateError` key in the `noise_params` argument.
It's important to note that, depending on the flag set in `Ansaetze.UnitaryGates.batch_gate_error`, the error will be applied to the entire batch of parameters (all parameters are affected in the same way) or to each parameter individually (default).
This can be particularly usefull in a scenario where one would like to apply noise e.g. only on the encoding gates but wants to change them all uniformly.
An example of this is provided in the following code:

```python
from qml_essentials.ansaetze import UnitaryGates

UnitaryGates.batch_gate_error = False
model(
    ...
    noise_params={
        "GateError": 0.01,
    }
)

def pqc_noise_free(*args, **kwargs):
    kwargs["noise_params"] = None
    return pqc(*args, **kwargs)
model.pqc = pqc_noise_free
```

> **Note:** When using a noisy circuit, make sure to run the model with the `density` execution type.

## Pulse Simulation

Our framework allows constructing circuits at the **pulse level**, where each gate is implemented as a time-dependent control pulse rather than an abstract unitary.  
This provides a more fine grained access to the simulation of the underlying physical process.
While we provide a developer-oriented overview in this section, we would like to highlight [Tilmann's Bachelor's Thesis](https://doi.org/10.5445/IR/1000184129) if you want to have a more detailled read into pulse-level simulation and quantum Fourier models.

We implement a fundamental set of gates (RX, RY, RZ, CZ) upon which other, more complex gates can be built.
The dependency graph is shown in the following figure:
![Dependency Graph](figures/pulse_gates_dependencies_light.png#only-light)
![Dependency Graph](figures/pulse_gates_dependencies_dark.png#only-dark)
In this graph, the edge weights represent the number child gates required to implement a particular gate.
The gates at the bottom represent the fundamental gates.

Generally, the gates are available through the same interface as the regular unitary gates.
Pulse simulation can easily be enabled by adding the `mode="pulse"` keyword argument, e.g.:

```python
from qml_essentials.ansaetze import Gates

Gates.CY(wires=[0, 1], mode="pulse")
```

### Pulse Parameters per Gate

You can use the `PulseInformation` class in `qml_essentials.ansaetze` to access both the number and optimized values of the pulse parameters for each gate.
Consider the following code snippet:

```python
from qml_essentials.ansaetze import PulseInformation as pinfo

gate = "CX"

print(f"Number of pulse parameters for {gate}: {pinfo.num_params(gate)}")
# Number of pulse parameters for CX: 9

gate_instance = pinfo.gate_by_name(gate)

print(f"Childs of {gate}: {gate_instance.childs}")
# Childs of CX: [H, CZ, H]

print(f"All parameters of {gate}: {len(gate_instance.params)}")
# All parameters of CX: 9

print(f"Leaf parameters of {gate}: {len(gate_instance.leaf_params)}")
# Leaf parameters of CX: 5
```

Looking back at the dependency graph, we can easily see where the discrepancy between the overall number parameters and the number of leaf parameters comes from.
The CX gate is composed of two Hadamard gates which in turn are decomposed into RY and RZ gates respectively.
By default, our implementation assumes, that you want to treat each rotational gate equally, thus the number of leaf parameters is just the "unique" number of parameter resulting after merging multiple occurencies of the same gate type.
However, it is also possible to overwrite these behavior, as we will see in the following example.

### Calling Gates in Pulse Mode

To execute a gate in pulse mode, provide `gate_mode="pulse"` when calling the gate.  
Optional `pulse_params` can be passed; if omitted, optimized default values are used:

```python
w = 3.14159

# CX gate with default optimized pulse parameters 
# (gates of equal type will recieve equal pulse parameters)
Gates.CX(w, wires=0, gate_mode="pulse")

# CX gate with custom pulse parameters (overwriting default pulse parameters)
pulse_params = [0.5, 7.9218643, 22.0381298, 1.09409231, 0.31830953, 0.5, 7.9218643, 22.0381298, 1.09409231]
Gates.RX(w, wires=0, gate_mode="pulse", pulse_params=pulse_params)
```

### Building Ansatzes in Pulse Mode

When building an ansatz in pulse mode (via a `Model`), the framework internally passes an array of ones as **element-wise scalers** for the optimized parameters.  
If `pulse_params` are provided for a model or gate, these are treated similarly as element-wise scalers to modify the default pulses. We again take advantage of the **kwargs and call:

```python
model(gate_mode="pulse", pulse_params=model.pulse_params_scaler * 1.5)
```

Here, input and params are inferred from the `Model` instance, and we scale all pulse parameters by a factor of 1.5.
Currently there is no way to change the raw values of pulse parameter through the model api directly.

> **Note:** Pulse-level simulation currently **does not support noise channels**. Mixing with noise will raise an error.  

### Quantum Optimal Control (QOC)

Our package provides a QOC interface for directly optimizing pulse parameters for specific gates.  

> **QOC is currently WIP, therefore only minimal documentation is provided**

Conceptually the provided QOC class contains methods to create test circuits (`create_GATE`) which return two circuits, one using the pulse level implementation of `GATE` and the other using the unitary level implementation of `GATE`.
For the specific implementation of these methods, we refer to the documentation of the `QOC` class.
To test a broad range of states, each of these circuits does not only include the `GATE` itself, but other, unitary based gates as well.
Those usually take a paramter `w`, allowing to sweep through the parameter space and validate if `GATE` acutally mimics its unitary counterpart.

Without any parameter specification, we can initialize the QOC class:

```python
from qml_essentials.qoc import QOC

qoc = QOC()
```

For a detailled description of available arguments, we refer to the documentation of the `QOC` class.
Now, we can select a gate of pass `sel_gates="all"` when calling `optimize_all`:

```python
qoc.optimize_all(sel_gates="GATE")
```

which will run the optimization for the specified gate.
The output of the optimization is logged to `qoc_logs.csv` whereas the resulting pulse parameters are stored in `qoc_results.csv`.
  
For further examples we refer to our ["Pulses" notebook](https://github.com/cirKITers/qml-essentials/blob/main/docs/pulses.ipynb) .

With the optimized pulse parameters we can generate a fidelities plot as follows:

![Gate Fidelities](figures/gates_fidelities_light.png#light-only)
![Gate Fidelities](figures/gates_fidelities_dark.png#dark-only)

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
