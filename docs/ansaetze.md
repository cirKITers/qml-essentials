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
from qml_essentials.ansaetze import Circuit, PulseInformation
from typing import Optional

class MyHardwareEfficient(Circuit):
    @staticmethod
    def n_params_per_layer(n_qubits: int) -> int:
        return n_qubits * 3

    @staticmethod
    def n_pulse_params_per_layer(n_qubits: int) -> int:
        n_params_RY = PulseInformation.num_params("RY")
        n_params_RZ = PulseInformation.num_params("RZ")
        n_params_CZ = PulseInformation.num_params("CZ")

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

To demonstrate this, let's recall the custom ansatz `MyHarwareEfficient` defined in [Custom Ansatz](#custom-ansatz) and extend the model's usage:

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

> **Note:** When using a noisy circuit, make sure to run the model with the `density` execution type.

## Pulse Simulation

Our framework allows constructing circuits at the **pulse level**, where each gate is implemented as a time-dependent control pulse rather than an abstract unitary.  
This provides a more realistic simulation of superconducting qubit hardware.  

### Pulse Parameters per Gate

Each implemented gate takes the following number of pulse parameters:

| Gate  | Pulse Parameters | Description |
|-------|-----------------|-------------|
| Rot   | -               | Not implemented |
| RX    | 3               | (A, σ, t) amplitude, width, and pulse duration |
| RY    | 3               | (A, σ, t) amplitude, width, and pulse duration |
| RZ    | 1               | t: pulse duration |
| CRX   | -               | Not implemented |
| CRY   | -               | Not implemented |
| CRZ   | -               | Not implemented |
| CX    | 4               | (A_H, σ_H, t_H, t_CZ) parameters for decomposed pulse sequence |
| CY    | -               | Not implemented |
| CZ    | 1               | t_CZ: pulse duration |
| H     | 3               | (A_H, σ_H, t_H) passed to underlying RY decomposition |

You can use the `PulseInformation` class to access both the number and optimized values of the pulse parameters for each gate:

```python
from qml_essentials.ansaetze import PulseInformation

# Number of pulse parameters
n_pulse_params_RX = PulseInformation.num_params("RX")  # 3
n_pulse_params_CX = PulseInformation.num_params("CX")  # 4

# Optimized pulse parameters
opt_pulse_params_RX = PulseInformation.optimized_params("RX")  # jnp.array([15.709893, 29.52306, 0.74998104])
opt_pulse_params_CX = PulseInformation.optimized_params("CX")  # jnp.array([7.9447253, 21.6398258, 0.90724313, 0.95509776])
```

### Calling Gates in Pulse Mode

To execute a gate in pulse mode, provide `gate_mode="pulse"` when calling the gate.  
Optional `pulse_params` can be passed; if omitted, optimized default values are used:

```python
w = 3.14159

# RX gate with default optimized pulse parameters
Gates.RX(w, wires=0, gate_mode="pulse")

# RX gate with custom pulse parameters
pulse_params = [0.5, 0.2, 1.0]  # A, σ, t
Gates.RX(w, wires=0, gate_mode="pulse", pulse_params=pulse_params)
```

### Building Ansatzes in Pulse Mode

When building an ansatz in pulse mode (via a `Model`), the framework internally passes an array of ones as **element-wise scalers** for the optimized parameters.  
If `pulse_params` are provided for a model or gate, these are treated similarly as element-wise scalers to modify the default pulses. We again take advantage of the **kwargs and call:

```python
model(model.params, inputs=None, gate_mode="pulse")
```

> **Note:** Pulse-level simulation currently **does not support noise channels**. Mixing with noise will raise an error.  

### Quantum Optimal Control (QOC)

Our package provides a QOC interface for directly optimizing pulse parameters for specific gates.  

#### QOC Class Initialization

The `QOC` class constructor supports the following arguments:

- `make_plots=False` — whether to generate and save plots after optimization  
- `file_dir="qoc/results"`- directory to save optimization results
- `fig_dir="qoc/figures"` — directory to save figures if `make_plots=True`  
- `fig_points=70` — number of points to plot for each pulse  

```python
from qml_essentials.qoc import QOC

# Initialize a QOC object
qoc = QOC(
    make_plots=True,
    file_dir="qoc",
    fig_dir="qoc",
    fig_points=100
)
```

#### Optimizing Pulse Parameters

The `optimize_*` functions (e.g., `optimize_RX`) optimize pulse parameters to best approximate a target unitary gate.  
For example, `optimize_RX` uses gradient-based optimization to minimize the difference between the pulse-based RX(w) circuit expectation value and the target gate-based RX(w).

Arguments:

- `steps: int = 1000` — maximum number of optimization steps  
- `patience: int = 100` — number of steps without improvement before early stopping  
- `w: float = jnp.pi` — target rotation angle  
- `init_pulse_params: jnp.array = jnp.array([1.0, 15.0, 1.0])` — initial guess for the pulse parameters (A, σ, t)  
- `print_every: int = 50` — print progress every N steps  

Returns:

- `tuple` — `(optimized_params, loss, losses)` where `optimized_params` is a `jnp.ndarray` of optimized pulse parameters (the best found during training), `loss` is the best loss, and `losses` is a list of loss values during optimization. Here, loss is defined as `1 - fidelity`.

Example usage:

```python
# Optimize pulse parameters for an RX rotation
optimized_pulse_params, best_loss, loss_values = qoc.optimize_RX(
    steps=2000,
    patience=200,
    w=jnp.pi/2,
    init_pulse_params=jnp.array([0.8, 10.0, 1.2]),
    print_every=100
)
print(f"Optimized parameters for RX: {optimized_pulse_params}\n")
print(f"Best achieved fidelity: {1 - best_loss}")

# Optimize pulse parameters for an RY rotation
optimized_pulse_params, best_loss, loss_values = qoc.optimize_RY()
print(f"Optimized parameters for RY: {optimized_pulse_params}\n")
print(f"Best achieved fidelity: {1 - best_loss:.6f}")
```

Currently, QOC is implemented for all qubit gates with pulse-level support (RX, RY, RZ, H, CX, CZ).  

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
