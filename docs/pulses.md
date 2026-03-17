# Pulses

Our framework allows constructing circuits at the **pulse level**, where each gate is implemented as a time-dependent control pulse rather than an abstract unitary.  
This provides a more fine grained access to the simulation of the underlying physical process.
While we provide a developer-oriented overview in this section, we would like to highlight [Tilmann's Bachelor's Thesis](https://doi.org/10.5445/IR/1000184129) if you want to have a more detailled read into pulse-level simulation and quantum Fourier models.

We implement a fundamental set of gates (RX, RY, RZ, CZ) upon which other, more complex gates can be built.
The dependency graph is shown in the following figure:
![Dependency Graph](figures/pulse_gates_dependencies_light.png#center#only-light)
![Dependency Graph](figures/pulse_gates_dependencies_dark.png#center#only-dark)
In this graph, the edge weights represent the number child gates required to implement a particular gate.
The gates at the bottom represent the fundamental gates.

Generally, the gates are available through the same interface as the regular unitary gates.
Pulse simulation can easily be enabled by adding the `mode="pulse"` keyword argument, e.g.:

```python
from qml_essentials.gates import Gates

Gates.CY(wires=[0, 1], mode="pulse")
```

## Pulse Parameters per Gate

You can use the `PulseInformation` class in `qml_essentials.ansaetze` to access both the number and optimized values of the pulse parameters for each gate.
Consider the following code snippet:

```python
from qml_essentials.gates import PulseInformation as pinfo

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

## Calling Gates in Pulse Mode

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

## Building Ansatzes in Pulse Mode

When building an ansatz in pulse mode (via a `Model`), the framework internally passes an array of ones as **element-wise scalers** for the optimized parameters.  
If `pulse_params` are provided for a model or gate, these are treated similarly as element-wise scalers to modify the default pulses. We again take advantage of the **kwargs and call:

```python
model(gate_mode="pulse", pulse_params=model.pulse_params_scaler * 1.5)
```

Here, input and params are inferred from the `Model` instance, and we scale all pulse parameters by a factor of 1.5.
Currently there is no way to change the raw values of pulse parameter through the model api directly.

Similar to the input and standard parameters, we also support batching for the `pulse_params` argument, meaning that you can also pass a batched array of pulse parameters of e.g. size 2 to as follows:

```python
model(pulse_params=np.repeat(model.pulse_params, 2, axis=-1), gate_mode="pulse")
``` 

## Quantum Optimal Control (QOC)

Our package provides a QOC interface for directly optimizing pulse parameters for specific gates.  
Conceptually the provided QOC class contains methods to create test circuits (`create_GATE`) which return two circuits, one using the pulse level implementation of `GATE` and the other using the unitary level implementation of `GATE`.
For the specific implementation of these methods, we refer to the documentation of the `QOC` class.
To test a broad range of states, each of these circuits does not only include the `GATE` itself, but other, unitary based gates as well.
Those usually take a paramter `w`, allowing to sweep through the parameter space and validate if `GATE` acutally mimics its unitary counterpart.

Using the standard parameter specification, we can initialize the QOC class:

```python
from qml_essentials.qoc import QOC, default_qoc_params

qoc = QOC(**default_qoc_params)
```

For a detailled description of available arguments, we refer to the documentation of the `QOC` class.
Now, we can select a gate of pass `sel_gates="GATE"` when calling `optimize_all`:

```python
qoc.optimize_all(sel_gates=["RX", "RY", "RZ", "CZ"])
```

which will run the optimization for the specified gate.
The output of the optimization is logged to `qoc_logs.csv` whereas the resulting pulse parameters are stored in `qoc_results.csv`.
  
Internally a multiobjective cost function is utilized to tune the pulse parameters of the basis gates.
Primarily, the fidelity between the pulse gate and a target unitary is optimized, but the default setting also takes into account the width of the pulse and a time normalization.
We refer to the exact weighting between these cost functions to the actual values in `default_qoc_params`.

Besides the cost function and their respective weight, you can also specify the envelope used for the pulse gate.

For further examples we refer to our ["Pulses" notebook](https://github.com/cirKITers/qml-essentials/blob/main/docs/pulses.ipynb) .

With the optimized pulse parameters we can generate a fidelities plot as follows:

![Gate Fidelities](figures/gates_fidelities_light.png#center#only-light)
![Gate Fidelities](figures/gates_fidelities_dark.png#center#only-dark)

Note that in this plot, the phase error is shown as $1-\text{phase error}$ to align it with the fidelity scale.
